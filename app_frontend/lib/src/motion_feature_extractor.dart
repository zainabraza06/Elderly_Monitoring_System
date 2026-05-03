import 'dart:math' as math;

import 'models.dart';

/// Colab / Python `scripts/baseline_fall/enhanced_features.py` — **116** floats per window.
class MotionFeatureExtractor {
  MotionFeatureExtractor._();

  /// Must match training (300 samples @ ~50 Hz).
  static const int windowLength = 300;
  static const int enhancedFeatureDim = 116;

  /// Build **116** features from one window of sensor readings.
  /// If [samples] has length ≠ 300, values are linearly resampled to 300 per axis.
  static List<double> extractEnhanced(List<SensorReadingPayload> samples) {
    if (samples.length < 2) {
      throw ArgumentError('Need at least 2 samples to form a window.');
    }

    final n = samples.length;
    final accX = List<double>.generate(n, (i) => samples[i].accX);
    final accY = List<double>.generate(n, (i) => samples[i].accY);
    final accZ = List<double>.generate(n, (i) => samples[i].accZ);
    final gX = List<double>.generate(n, (i) => samples[i].gyroX);
    final gY = List<double>.generate(n, (i) => samples[i].gyroY);
    final gZ = List<double>.generate(n, (i) => samples[i].gyroZ);

    final ax = n == windowLength ? accX : _resampleSeries(accX, windowLength);
    final ay = n == windowLength ? accY : _resampleSeries(accY, windowLength);
    final az = n == windowLength ? accZ : _resampleSeries(accZ, windowLength);
    final gx = n == windowLength ? gX : _resampleSeries(gX, windowLength);
    final gy = n == windowLength ? gY : _resampleSeries(gY, windowLength);
    final gz = n == windowLength ? gZ : _resampleSeries(gZ, windowLength);

    // Orientation unknown on phone — zeros (same as missing ori in CSV).
    final oz = List<double>.filled(windowLength, 0.0);
    final ori = <List<double>>[
      List<double>.from(oz),
      List<double>.from(oz),
      List<double>.from(oz),
    ];

    final acc = <List<double>>[ax, ay, az];
    final gyro = <List<double>>[gx, gy, gz];

    final feat = <double>[];

    for (var axis = 0; axis < 3; axis++) {
      final data = acc[axis];
      feat.addAll(_accAxisTimeFreq(data));
    }

    if (_anyNonZero(gyro[0]) || _anyNonZero(gyro[1]) || _anyNonZero(gyro[2])) {
      for (var axis = 0; axis < 3; axis++) {
        final data = gyro[axis];
        feat.addAll(_gyroStats(data));
      }
    } else {
      feat.addAll(List<double>.filled(15, 0.0));
    }

    if (_anyNonZero(ori[0]) || _anyNonZero(ori[1]) || _anyNonZero(ori[2])) {
      for (var axis = 0; axis < 3; axis++) {
        final data = ori[axis];
        final d0 = data[0];
        final d1 = data[data.length - 1];
        feat.addAll([_mean(data), _std(data), d1 - d0]);
      }
    } else {
      feat.addAll(List<double>.filled(9, 0.0));
    }

    feat.add(_corr(acc[0], acc[1]));
    feat.add(_corr(acc[0], acc[2]));
    feat.add(_corr(acc[1], acc[2]));

    final mag = List<double>.generate(
      windowLength,
      (i) => math.sqrt(acc[0][i] * acc[0][i] + acc[1][i] * acc[1][i] + acc[2][i] * acc[2][i]),
    );
    feat.addAll(_magnitudeBlock(mag));

    if (feat.length != enhancedFeatureDim) {
      throw StateError('Feature length ${feat.length} != $enhancedFeatureDim');
    }
    return feat;
  }

  static List<double> _resampleSeries(List<double> src, int targetLen) {
    if (src.length == targetLen) return List<double>.from(src);
    final out = <double>[];
    final last = src.length - 1;
    for (var t = 0; t < targetLen; t++) {
      final u = last * t / (targetLen - 1);
      final i = u.floor();
      final j = (i + 1).clamp(0, last);
      final f = u - i;
      out.add(src[i] * (1 - f) + src[j] * f);
    }
    return out;
  }

  static bool _anyNonZero(List<double> v) {
    for (final x in v) {
      if (x != 0.0) return true;
    }
    return false;
  }

  static List<double> _accAxisTimeFreq(List<double> data) {
    final out = <double>[];
    out.addAll([
      _mean(data),
      _std(data),
      _median(data),
      data.reduce(math.min),
      data.reduce(math.max),
      _ptp(data),
      _percentile(data, 5),
      _percentile(data, 25),
      _percentile(data, 75),
      _percentile(data, 95),
      math.sqrt(data.map((e) => e * e).reduce((a, b) => a + b) / data.length),
      _meanAbsDiff(data),
      _sumAbsDiff(data),
      _skew(data),
      _kurtosisExcess(data),
      _variance(data),
      data.map((e) => e * e).reduce((a, b) => a + b) / data.length,
      data.map((e) => e.abs()).reduce(math.max),
      _argmaxAbs(data) / data.length,
    ]);

    final half = _rfftMagnitudes(data);
    if (half.isEmpty) {
      out.addAll(List<double>.filled(6, 0.0));
    } else {
      final s = half.fold<double>(0.0, (a, b) => a + b) + 1e-6;
      final low = half.length >= 10 ? half.sublist(0, 10).fold<double>(0.0, (a, b) => a + b) : s;
      out.addAll([
        _mean(half),
        _std(half),
        half.reduce(math.max),
        s,
        half.indexOf(half.reduce(math.max)) / half.length,
        low / s,
      ]);
    }

    if (data.length > 1) {
      var zc = 0;
      for (var i = 0; i < data.length - 1; i++) {
        if (_signNum(data[i]) != _signNum(data[i + 1])) zc++;
      }
      out.add(zc / data.length);
    } else {
      out.add(0.0);
    }
    return out;
  }

  static List<double> _gyroStats(List<double> data) {
    final ms = data.map((e) => e.abs());
    final sumAbs = ms.fold<double>(0.0, (a, b) => a + b);
    return [
      _mean(data),
      _std(data),
      ms.reduce(math.max),
      sumAbs,
      data.map((e) => e * e).reduce((a, b) => a + b) / data.length,
    ];
  }

  static List<double> _magnitudeBlock(List<double> magnitude) {
    final out = <double>[
      _mean(magnitude),
      _std(magnitude),
      magnitude.reduce(math.max),
      _percentile(magnitude, 95),
      _argmax(magnitude) / magnitude.length,
      magnitude.fold<double>(0.0, (a, b) => a + b),
      _meanAbsDiff(magnitude),
    ];
    if (magnitude.length > 10) {
      final h = _std(magnitude);
      final peaks = _findPeaks(magnitude, height: h, distance: 5);
      if (peaks.isEmpty) {
        out.addAll([0.0, 0.0, 0.0, 0.0]);
      } else {
        final ph = peaks.map((i) => magnitude[i]).toList();
        out.addAll([
          peaks.length.toDouble(),
          ph.reduce(math.max),
          _mean(ph),
          peaks.first / magnitude.length,
        ]);
      }
    } else {
      out.addAll([0.0, 0.0, 0.0, 0.0]);
    }
    return out;
  }

  /// scipy.signal.find_peaks with scalar `height` and `distance` (approximate).
  static List<int> _findPeaks(List<double> y, {required double height, required int distance}) {
    final n = y.length;
    final candidates = <int>[];
    for (var i = 0; i < n; i++) {
      if (y[i] < height) continue;
      var isLocalMax = true;
      if (i > 0 && y[i] < y[i - 1]) isLocalMax = false;
      if (i < n - 1 && y[i] < y[i + 1]) isLocalMax = false;
      if (isLocalMax) candidates.add(i);
    }
    candidates.sort((a, b) => y[b].compareTo(y[a]));
    final picked = <int>[];
    for (final idx in candidates) {
      if (picked.every((p) => (p - idx).abs() >= distance)) {
        picked.add(idx);
      }
    }
    picked.sort();
    return picked;
  }

  static int _signNum(double x) => x > 0 ? 1 : (x < 0 ? -1 : 0);

  static int _argmax(List<double> d) {
    var best = 0;
    for (var i = 1; i < d.length; i++) {
      if (d[i] > d[best]) best = i;
    }
    return best;
  }

  static int _argmaxAbs(List<double> d) {
    var best = 0;
    for (var i = 1; i < d.length; i++) {
      if (d[i].abs() > d[best].abs()) best = i;
    }
    return best;
  }

  /// Naive DFT magnitudes for indices 0 .. n/2 - 1 (matches numpy fft ordering roughly).
  static List<double> _rfftMagnitudes(List<double> x) {
    final n = x.length;
    final half = n ~/ 2;
    if (half == 0) return [];
    final out = List<double>.filled(half, 0.0);
    const tau = 2 * math.pi;
    for (var k = 0; k < half; k++) {
      double re = 0, im = 0;
      for (var t = 0; t < n; t++) {
        final angle = -tau * k * t / n;
        re += x[t] * math.cos(angle);
        im += x[t] * math.sin(angle);
      }
      out[k] = math.sqrt(re * re + im * im);
    }
    return out;
  }

  static double _mean(List<double> d) =>
      d.isEmpty ? 0.0 : d.fold<double>(0.0, (a, b) => a + b) / d.length;

  static double _variance(List<double> d) {
    if (d.length < 2) return 0.0;
    final m = _mean(d);
    return d.map((v) => (v - m) * (v - m)).reduce((a, b) => a + b) / d.length;
  }

  static double _std(List<double> d) => math.sqrt(_variance(d));

  static double _median(List<double> d) {
    final s = List<double>.from(d)..sort();
    final n = s.length;
    if (n == 0) return 0.0;
    if (n.isOdd) return s[n ~/ 2];
    return (s[n ~/ 2 - 1] + s[n ~/ 2]) / 2;
  }

  static double _ptp(List<double> d) =>
      d.isEmpty ? 0.0 : d.reduce(math.max) - d.reduce(math.min);

  static double _percentile(List<double> d, int p) {
    if (d.isEmpty) return 0.0;
    final s = List<double>.from(d)..sort();
    final idx = (p / 100.0) * (s.length - 1);
    final lo = idx.floor();
    final hi = idx.ceil().clamp(0, s.length - 1);
    final f = idx - lo;
    return s[lo] * (1 - f) + s[hi] * f;
  }

  static double _meanAbsDiff(List<double> d) {
    if (d.length < 2) return 0.0;
    var sum = 0.0;
    for (var i = 0; i < d.length - 1; i++) {
      sum += (d[i + 1] - d[i]).abs();
    }
    return sum / (d.length - 1);
  }

  static double _sumAbsDiff(List<double> d) {
    if (d.length < 2) return 0.0;
    var sum = 0.0;
    for (var i = 0; i < d.length - 1; i++) {
      sum += (d[i + 1] - d[i]).abs();
    }
    return sum;
  }

  static double _skew(List<double> d) {
    if (d.length < 3) return 0.0;
    final m = _mean(d);
    var m2 = 0.0, m3 = 0.0;
    for (final v in d) {
      final x = v - m;
      m2 += x * x;
      m3 += x * x * x;
    }
    m2 /= d.length;
    m3 /= d.length;
    if (m2 < 1e-12) return 0.0;
    return m3 / (m2 * math.sqrt(m2));
  }

  /// scipy.stats.kurtosis (Fisher / excess).
  static double _kurtosisExcess(List<double> d) {
    if (d.length < 4) return 0.0;
    final m = _mean(d);
    var m2 = 0.0, m4 = 0.0;
    for (final v in d) {
      final x = v - m;
      m2 += x * x;
      m4 += x * x * x * x;
    }
    m2 /= d.length;
    m4 /= d.length;
    if (m2 < 1e-12) return 0.0;
    return m4 / (m2 * m2) - 3.0;
  }

  static double _corr(List<double> a, List<double> b) {
    if (a.length != b.length || a.isEmpty) return 0.0;
    final ma = _mean(a);
    final mb = _mean(b);
    var num = 0.0, da = 0.0, db = 0.0;
    for (var i = 0; i < a.length; i++) {
      final xa = a[i] - ma;
      final xb = b[i] - mb;
      num += xa * xb;
      da += xa * xa;
      db += xb * xb;
    }
    final den = math.sqrt(da * db);
    if (den < 1e-12) return 0.0;
    final c = num / den;
    return c.isNaN ? 0.0 : c;
  }
}
