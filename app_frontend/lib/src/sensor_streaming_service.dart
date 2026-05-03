import 'dart:async';
import 'dart:math' as math;

import 'package:motion_core/motion_core.dart';
import 'package:sensors_plus/sensors_plus.dart';

import 'models.dart';

typedef SensorBatchCallback =
    Future<void> Function(List<SensorReadingPayload> samples);

/// Convert fused quaternion attitude to MobiAct CSV orientation (degrees).
/// MobiAct `ori`: Azimuth(Z), Pitch(X), Roll(Y). [MotionDataEuler]: yaw(Z), pitch(Y), roll(X) in radians.
({double azimuthDeg, double pitchDeg, double rollDeg}) mobiActOrientationFromMotion(
  MotionData m,
) {
  final yaw = m.yaw;
  final pitchY = m.pitch;
  final rollX = m.roll;
  final radToDeg = 180.0 / math.pi;
  return (
    azimuthDeg: _normalizeAzimuthDeg(yaw * radToDeg),
    pitchDeg: rollX * radToDeg,
    rollDeg: pitchY * radToDeg,
  );
}

double _normalizeAzimuthDeg(double d) {
  var x = d % 360.0;
  if (x < 0) x += 360.0;
  return x;
}

class SensorStreamingService {
  SensorStreamingService({
    this.targetSamplingRateHz = 50.0,
    this.windowSize = 128,
    this.stepSize = 64,
  });

  final double targetSamplingRateHz;
  final int windowSize;
  final int stepSize;

  final List<SensorReadingPayload> _buffer = <SensorReadingPayload>[];

  StreamSubscription<AccelerometerEvent>? _accelerometerSubscription;
  StreamSubscription<GyroscopeEvent>? _gyroscopeSubscription;
  StreamSubscription<MotionData>? _motionSubscription;

  double _latestGyroX = 0.0;
  double _latestGyroY = 0.0;
  double _latestGyroZ = 0.0;
  double? _latestAzimuthDeg;
  double? _latestPitchDeg;
  double? _latestRollDeg;

  int _lastSampleTimestampMs = 0;
  bool _isFlushing = false;
  bool _isRunning = false;

  bool get isRunning => _isRunning;
  int get bufferedSamples => _buffer.length;

  Future<SensorAccessStatus> probeSensors({
    Duration timeout = const Duration(seconds: 2),
  }) async {
    final accelerometerAvailable = await _checkStreamAvailable(
      accelerometerEventStream(),
      timeout,
    );
    final gyroscopeAvailable = await _checkStreamAvailable(
      gyroscopeEventStream(),
      timeout,
    );

    var fusedOrientationAvailable = false;
    try {
      fusedOrientationAvailable = await MotionCore.isAvailable();
    } catch (_) {
      fusedOrientationAvailable = false;
    }

    return SensorAccessStatus(
      accelerometerAvailable: accelerometerAvailable,
      gyroscopeAvailable: gyroscopeAvailable,
      fusedOrientationAvailable: fusedOrientationAvailable,
      checkedAt: DateTime.now(),
    );
  }

  Future<void> start(SensorBatchCallback onBatch) async {
    if (_isRunning) {
      return;
    }

    _buffer.clear();
    _lastSampleTimestampMs = 0;
    _latestAzimuthDeg = null;
    _latestPitchDeg = null;
    _latestRollDeg = null;
    _isRunning = true;

    _gyroscopeSubscription = gyroscopeEventStream().listen(
      (event) {
        _latestGyroX = event.x;
        _latestGyroY = event.y;
        _latestGyroZ = event.z;
      },
      onError: (_) {
        _latestGyroX = 0.0;
        _latestGyroY = 0.0;
        _latestGyroZ = 0.0;
      },
      cancelOnError: false,
    );

    try {
      if (await MotionCore.isAvailable()) {
        _motionSubscription = MotionCore.motionStream.listen(
          (MotionData data) {
            if (!_isRunning) return;
            final o = mobiActOrientationFromMotion(data);
            _latestAzimuthDeg = o.azimuthDeg;
            _latestPitchDeg = o.pitchDeg;
            _latestRollDeg = o.rollDeg;
          },
          onError: (_) {},
          cancelOnError: false,
        );
      }
    } catch (_) {}

    _accelerometerSubscription = accelerometerEventStream().listen(
      (event) {
        final nowMs = DateTime.now().millisecondsSinceEpoch;
        final minGapMs = (1000 / targetSamplingRateHz).round();
        if (_lastSampleTimestampMs != 0 &&
            nowMs - _lastSampleTimestampMs < minGapMs) {
          return;
        }
        _lastSampleTimestampMs = nowMs;

        _buffer.add(
          SensorReadingPayload(
            timestampMs: nowMs,
            accX: event.x,
            accY: event.y,
            accZ: event.z,
            gyroX: _latestGyroX,
            gyroY: _latestGyroY,
            gyroZ: _latestGyroZ,
            azimuth: _latestAzimuthDeg,
            pitch: _latestPitchDeg,
            roll: _latestRollDeg,
          ),
        );

        if (_buffer.length >= windowSize) {
          _flush(onBatch);
        }
      },
      onError: (_) {
        _isRunning = false;
      },
      cancelOnError: true,
    );
  }

  Future<void> stop() async {
    _isRunning = false;
    await _motionSubscription?.cancel();
    _motionSubscription = null;
    await _accelerometerSubscription?.cancel();
    await _gyroscopeSubscription?.cancel();
    _accelerometerSubscription = null;
    _gyroscopeSubscription = null;
    _buffer.clear();
  }

  void _flush(SensorBatchCallback onBatch) {
    if (_isFlushing || _buffer.length < windowSize) {
      return;
    }

    _isFlushing = true;
    final batch = List<SensorReadingPayload>.from(_buffer.take(windowSize));
    final overlapStep = stepSize < 1
        ? 1
        : (stepSize > windowSize ? windowSize : stepSize);
    _buffer.removeRange(0, overlapStep);

    onBatch(batch).whenComplete(() {
      _isFlushing = false;
      if (_isRunning && _buffer.length >= windowSize) {
        _flush(onBatch);
      }
    });
  }

  Future<bool> _checkStreamAvailable<T>(
    Stream<T> stream,
    Duration timeout,
  ) async {
    final completer = Completer<bool>();
    StreamSubscription<T>? subscription;
    Timer? timer;

    void finish(bool value) {
      if (completer.isCompleted) {
        return;
      }
      completer.complete(value);
      timer?.cancel();
      subscription?.cancel();
    }

    subscription = stream.listen(
      (_) => finish(true),
      onError: (_) => finish(false),
      cancelOnError: true,
    );

    timer = Timer(timeout, () => finish(false));
    return completer.future;
  }
}
