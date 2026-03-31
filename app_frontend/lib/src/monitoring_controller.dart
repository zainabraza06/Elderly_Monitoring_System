import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api_client.dart';
import 'models.dart';
import 'sensor_streaming_service.dart';

class MonitoringController extends ChangeNotifier {
  MonitoringController({
    BackendApiClient? apiClient,
    SensorStreamingService? sensorService,
  })  : _apiClient = apiClient ?? BackendApiClient(baseUrl: defaultBackendUrl),
        _sensorService = sensorService ??
            SensorStreamingService(
              targetSamplingRateHz: defaultSampleRateHz,
              windowSize: offlineWindowSizeSamples,
              stepSize: offlineWindowStepSamples,
            );

  static const String defaultBackendUrl = 'http://10.0.2.2:8000';
  static const String defaultDeviceLabel = 'Caregiver Phone';
  static const double defaultSampleRateHz = 50.0;
  static const int offlineWindowSizeSamples = 128;
  static const int offlineWindowStepSamples = 64;

  static const String _backendUrlKey = 'backend_url';
  static const String _patientNameKey = 'patient_name';
  static const String _patientAgeKey = 'patient_age';
  static const String _roomLabelKey = 'room_label';
  static const String _deviceLabelKey = 'device_label';
  static const String _patientIdKey = 'patient_id';
  static const String _deviceIdKey = 'device_id';

  final BackendApiClient _apiClient;
  final SensorStreamingService _sensorService;

  SharedPreferences? _preferences;

  bool _initialized = false;
  bool _isBusy = false;
  bool _backendReachable = false;
  bool _isStreaming = false;

  String _backendUrl = defaultBackendUrl;
  String _patientName = '';
  int? _patientAge;
  String _roomLabel = '';
  String _deviceLabel = defaultDeviceLabel;

  String? _patientId;
  String? _deviceId;
  String? _sessionId;

  String _statusMessage = 'Enter patient and backend details to begin.';
  String? _lastError;

  int _batchesSent = 0;
  int _lastBatchSize = 0;
  DateTime? _lastTransmissionAt;

  DetectionResultModel? _lastDetection;
  LiveStatusModel? _liveStatus;
  AlertRecordModel? _activeAlert;
  TelemetrySnapshotModel? _latestTelemetry;
  SensorAccessStatus? _sensorAccessStatus;

  bool get initialized => _initialized;
  bool get isBusy => _isBusy;
  bool get backendReachable => _backendReachable;
  bool get isStreaming => _isStreaming;
  bool get hasSetup =>
      _backendUrl.trim().isNotEmpty &&
      _patientName.trim().isNotEmpty &&
      _deviceLabel.trim().isNotEmpty;
  bool get isReady => hasSetup && _backendReachable;

  String get backendUrl => _backendUrl;
  String get patientName => _patientName;
  int? get patientAge => _patientAge;
  String get roomLabel => _roomLabel;
  String get deviceLabel => _deviceLabel;
  String? get patientId => _patientId;
  String? get deviceId => _deviceId;
  String? get sessionId => _sessionId;
  String get statusMessage => _statusMessage;
  String? get lastError => _lastError;
  int get batchesSent => _batchesSent;
  int get lastBatchSize => _lastBatchSize;
  DateTime? get lastTransmissionAt => _lastTransmissionAt;
  DetectionResultModel? get lastDetection => _lastDetection;
  LiveStatusModel? get liveStatus => _liveStatus;
  AlertRecordModel? get activeAlert => _activeAlert;
  TelemetrySnapshotModel? get latestTelemetry => _latestTelemetry;
  SensorAccessStatus? get sensorAccessStatus => _sensorAccessStatus;

  Future<void> initialize() async {
    if (_initialized) {
      return;
    }

    _preferences = await SharedPreferences.getInstance();
    _backendUrl = _preferences?.getString(_backendUrlKey) ?? defaultBackendUrl;
    _patientName = _preferences?.getString(_patientNameKey) ?? '';
    _patientAge = _preferences?.getInt(_patientAgeKey);
    _roomLabel = _preferences?.getString(_roomLabelKey) ?? '';
    _deviceLabel = _preferences?.getString(_deviceLabelKey) ?? defaultDeviceLabel;
    _patientId = _preferences?.getString(_patientIdKey);
    _deviceId = _preferences?.getString(_deviceIdKey);
    _sessionId = null;

    _apiClient.updateBaseUrl(_backendUrl);

    _statusMessage = hasSetup
        ? 'Saved setup loaded. Check the backend and start monitoring.'
        : 'Enter patient and backend details to begin.';

    _initialized = true;
    notifyListeners();
  }

  Future<SensorAccessStatus> refreshSensorStatus({bool silent = false}) async {
    await _ensureInitialized();

    if (!silent) {
      _isBusy = true;
      _lastError = null;
      _statusMessage = 'Checking phone sensors...';
      notifyListeners();
    }

    try {
      final status = await _sensorService.probeSensors();
      _sensorAccessStatus = status;
      if (!silent) {
        _statusMessage = status.allAvailable
            ? 'Phone sensors are available and ready.'
            : 'Some required sensors are unavailable on this device.';
      }
      return status;
    } catch (error) {
      final fallback = SensorAccessStatus(
        accelerometerAvailable: false,
        gyroscopeAvailable: false,
        checkedAt: DateTime.now(),
      );
      _sensorAccessStatus = fallback;
      _lastError = _formatError(error);
      if (!silent) {
        _statusMessage = 'Unable to verify phone sensors.';
      }
      return fallback;
    } finally {
      if (!silent) {
        _isBusy = false;
        notifyListeners();
      }
    }
  }

  Future<void> saveSetup({
    required String backendUrl,
    required String patientName,
    required String patientAgeText,
    required String roomLabel,
    required String deviceLabel,
  }) async {
    await _ensureInitialized();

    final normalizedBackendUrl =
        backendUrl.trim().isEmpty ? defaultBackendUrl : backendUrl.trim();
    final normalizedPatientName = patientName.trim();
    final normalizedRoomLabel = roomLabel.trim();
    final normalizedDeviceLabel =
        deviceLabel.trim().isEmpty ? defaultDeviceLabel : deviceLabel.trim();

    int? parsedAge;
    final trimmedAge = patientAgeText.trim();
    if (trimmedAge.isNotEmpty) {
      parsedAge = int.tryParse(trimmedAge);
      if (parsedAge == null || parsedAge < 0 || parsedAge > 130) {
        _lastError = 'Patient age must be a whole number between 0 and 130.';
        notifyListeners();
        return;
      }
    }

    final patientChanged = _patientName != normalizedPatientName ||
        _patientAge != parsedAge ||
        _roomLabel != normalizedRoomLabel;
    final deviceChanged = _deviceLabel != normalizedDeviceLabel;

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Saving setup...';
    notifyListeners();

    try {
      _backendUrl = normalizedBackendUrl;
      _patientName = normalizedPatientName;
      _patientAge = parsedAge;
      _roomLabel = normalizedRoomLabel;
      _deviceLabel = normalizedDeviceLabel;

      if (patientChanged) {
        _patientId = null;
        _sessionId = null;
        _lastDetection = null;
        _liveStatus = null;
        _activeAlert = null;
        _latestTelemetry = null;
      }

      if (deviceChanged) {
        _deviceId = null;
        _sessionId = null;
      }

      _apiClient.updateBaseUrl(_backendUrl);

      await _persistSetup();
      await _persistIdentifiers();

      final reachable = await refreshBackendReachability(silent: true);
      _statusMessage = reachable
          ? 'Setup saved. Backend connection looks healthy.'
          : 'Setup saved. Backend could not be reached yet.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Setup was updated locally, but the connection check failed.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<bool> refreshBackendReachability({bool silent = false}) async {
    await _ensureInitialized();

    if (!silent) {
      _isBusy = true;
      _lastError = null;
      _statusMessage = 'Checking backend connection...';
      notifyListeners();
    }

    try {
      await _apiClient.ping();
      _backendReachable = true;
      _lastError = null;
      if (!silent) {
        _statusMessage = 'Backend connection is healthy.';
      }
      return true;
    } catch (error) {
      _backendReachable = false;
      _lastError = _formatError(error);
      if (!silent) {
        _statusMessage = 'Backend is not reachable right now.';
      }
      return false;
    } finally {
      if (!silent) {
        _isBusy = false;
        notifyListeners();
      }
    }
  }

  Future<void> startMonitoring() async {
    await _ensureInitialized();

    if (_isStreaming) {
      return;
    }

    if (!hasSetup) {
      _lastError = 'Complete the backend URL, patient name, and device label first.';
      _statusMessage = 'Setup is incomplete.';
      notifyListeners();
      return;
    }

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Starting live monitoring...';
    notifyListeners();

    try {
      final reachable = await refreshBackendReachability(silent: true);
      if (!reachable) {
        throw ApiException(
          'The backend is not reachable. Verify the server is running and the URL is correct.',
        );
      }

      final sensorStatus = await refreshSensorStatus(silent: true);
      if (!sensorStatus.allAvailable) {
        throw ApiException(
          'Required phone sensors are not available. Check accelerometer and gyroscope access.',
        );
      }

      await _ensurePatient();
      await _ensureDevice();

      final session = await _apiClient.startSession(
        patientId: _patientId!,
        deviceId: _deviceId!,
        sampleRateHz: defaultSampleRateHz,
      );

      _sessionId = session.id;
      _isStreaming = true;
      _batchesSent = 0;
      _lastBatchSize = 0;
      _lastTransmissionAt = null;
      _lastDetection = null;
      _activeAlert = null;
      _latestTelemetry = null;
      _liveStatus = LiveStatusModel(
        patientId: _patientId!,
        patientName: _patientName,
        roomLabel: _roomLabel.isEmpty ? null : _roomLabel,
        sessionId: _sessionId,
        deviceId: _deviceId,
        severity: 'low',
        score: 0.0,
        fallProbability: 0.0,
        lastMessage: 'Session started. Waiting for live motion data...',
      );

      await _persistIdentifiers();
      _sensorService.start(_handleSensorBatch);
      _statusMessage = 'Monitoring is live. The phone is now streaming sensor batches.';
    } catch (error) {
      _isStreaming = false;
      _sessionId = null;
      _lastError = _formatError(error);
      _statusMessage = 'Unable to start monitoring.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> stopMonitoring() async {
    await _ensureInitialized();

    if (!_isStreaming && _sessionId == null) {
      return;
    }

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Stopping monitoring...';
    notifyListeners();

    final currentSessionId = _sessionId;

    try {
      await _sensorService.stop();
      _isStreaming = false;

      if (currentSessionId != null) {
        await _apiClient.stopSession(currentSessionId);
      }

      _sessionId = null;
      await _persistIdentifiers();
      _statusMessage = 'Monitoring stopped.';
    } catch (error) {
      _isStreaming = false;
      _sessionId = null;
      await _persistIdentifiers();
      _lastError = _formatError(error);
      _statusMessage =
          'Streaming stopped on the phone, but the backend session may still be open.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> triggerEmergencyAlert() async {
    await _ensureInitialized();

    if (_patientName.trim().isEmpty) {
      _lastError = 'Add a patient name before sending an emergency alert.';
      _statusMessage = 'Emergency alert was not sent.';
      notifyListeners();
      return;
    }

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Triggering emergency alert...';
    notifyListeners();

    try {
      final reachable = await refreshBackendReachability(silent: true);
      if (!reachable) {
        throw ApiException(
          'The backend is not reachable. The manual alert could not be delivered.',
        );
      }

      await _ensurePatient();
      await _ensureDevice();

      final alert = await _apiClient.triggerManualAlert(
        patientId: _patientId!,
        deviceId: _deviceId,
        sessionId: _sessionId,
      );

      _activeAlert = alert;
      _statusMessage = 'Emergency alert sent successfully.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Emergency alert failed.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> _handleSensorBatch(List<SensorReadingPayload> samples) async {
    if (!_isStreaming ||
        _patientId == null ||
        _deviceId == null ||
        _sessionId == null ||
        samples.isEmpty) {
      return;
    }

    final currentSessionId = _sessionId;

    try {
      final response = await _apiClient.ingestLiveBatch(
        patientId: _patientId!,
        deviceId: _deviceId!,
        sessionId: currentSessionId!,
        samplingRateHz: defaultSampleRateHz,
        batteryLevel: null,
        samples: samples,
      );

      if (!_isStreaming || _sessionId != currentSessionId) {
        return;
      }

      _backendReachable = true;
      _lastError = null;
      _batchesSent += 1;
      _lastBatchSize = samples.length;
      _lastTransmissionAt = DateTime.now();
      _lastDetection = response.detection;
      _liveStatus = response.liveStatus;
      _latestTelemetry = response.telemetry;

      if (response.activeAlert != null) {
        _activeAlert = response.activeAlert;
      } else if (response.liveStatus.activeAlertIds.isEmpty) {
        _activeAlert = null;
      }

      _statusMessage = response.detection.message;
    } catch (error) {
      if (!_isStreaming || _sessionId != currentSessionId) {
        return;
      }

      _backendReachable = false;
      _lastError = _formatError(error);
      _lastBatchSize = samples.length;
      _statusMessage =
          'Streaming is still active on the phone, but the latest batch upload failed.';
    } finally {
      notifyListeners();
    }
  }

  Future<void> _ensurePatient() async {
    if (_patientId != null) {
      try {
        await _apiClient.getPatient(_patientId!);
        return;
      } catch (_) {
        _patientId = null;
      }
    }

    final patient = await _apiClient.createPatient(
      fullName: _patientName,
      age: _patientAge,
      roomLabel: _roomLabel.isEmpty ? null : _roomLabel,
    );

    _patientId = patient.id;
    await _persistIdentifiers();
  }

  Future<void> _ensureDevice() async {
    if (_deviceId != null) {
      try {
        await _apiClient.getDevice(_deviceId!);
        return;
      } catch (_) {
        _deviceId = null;
      }
    }

    final device = await _apiClient.createDevice(
      label: _deviceLabel,
      ownerName: _patientName,
    );

    _deviceId = device.id;
    await _persistIdentifiers();
  }

  Future<void> _persistSetup() async {
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;

    await preferences.setString(_backendUrlKey, _backendUrl);
    await preferences.setString(_patientNameKey, _patientName);
    await preferences.setString(_roomLabelKey, _roomLabel);
    await preferences.setString(_deviceLabelKey, _deviceLabel);

    if (_patientAge == null) {
      await preferences.remove(_patientAgeKey);
    } else {
      await preferences.setInt(_patientAgeKey, _patientAge!);
    }
  }

  Future<void> _persistIdentifiers() async {
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;

    if (_patientId == null) {
      await preferences.remove(_patientIdKey);
    } else {
      await preferences.setString(_patientIdKey, _patientId!);
    }

    if (_deviceId == null) {
      await preferences.remove(_deviceIdKey);
    } else {
      await preferences.setString(_deviceIdKey, _deviceId!);
    }
  }

  Future<void> _ensureInitialized() async {
    if (!_initialized) {
      await initialize();
    }
  }

  String _formatError(Object error) {
    if (error is ApiException) {
      return error.message;
    }

    final message = error.toString();
    if (message.startsWith('Exception: ')) {
      return message.substring('Exception: '.length);
    }
    return message;
  }

  @override
  void dispose() {
    unawaited(_sensorService.stop());
    super.dispose();
  }
}
