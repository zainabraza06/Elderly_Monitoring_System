import 'dart:async';

import 'package:sensors_plus/sensors_plus.dart';

import 'models.dart';


typedef SensorBatchCallback = Future<void> Function(List<SensorReadingPayload> samples);


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

  double _latestGyroX = 0.0;
  double _latestGyroY = 0.0;
  double _latestGyroZ = 0.0;
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

    return SensorAccessStatus(
      accelerometerAvailable: accelerometerAvailable,
      gyroscopeAvailable: gyroscopeAvailable,
      checkedAt: DateTime.now(),
    );
  }

  void start(SensorBatchCallback onBatch) {
    if (_isRunning) {
      return;
    }

    _buffer.clear();
    _lastSampleTimestampMs = 0;
    _isRunning = true;

    _gyroscopeSubscription = gyroscopeEventStream().listen((event) {
      _latestGyroX = event.x;
      _latestGyroY = event.y;
      _latestGyroZ = event.z;
    });

    _accelerometerSubscription = accelerometerEventStream().listen((event) {
      final nowMs = DateTime.now().millisecondsSinceEpoch;
      final minGapMs = (1000 / targetSamplingRateHz).round();
      if (_lastSampleTimestampMs != 0 && nowMs - _lastSampleTimestampMs < minGapMs) {
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
        ),
      );

      if (_buffer.length >= windowSize) {
        _flush(onBatch);
      }
    });
  }

  Future<void> stop() async {
    _isRunning = false;
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
