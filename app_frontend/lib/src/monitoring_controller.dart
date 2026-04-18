import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api_client.dart';
import 'models.dart';
import 'sensor_streaming_service.dart';

class MonitoringController extends ChangeNotifier {
  MonitoringController({
    BackendApiClient? apiClient,
    SensorStreamingService? sensorService,
  })  : _apiClient = apiClient ?? BackendApiClient(),
        _sensorService = sensorService ??
            SensorStreamingService(
              targetSamplingRateHz: defaultSampleRateHz,
              windowSize: offlineWindowSizeSamples,
              stepSize: offlineWindowStepSamples,
            );

  static const String defaultDeviceLabel = 'Caregiver Phone';
  static const double defaultSampleRateHz = 50.0;
  static const int offlineWindowSizeSamples = 128;
  static const int offlineWindowStepSamples = 64;

  static const String _patientNameKey = 'patient_name';
  static const String _patientAgeKey = 'patient_age';
  static const String _roomLabelKey = 'room_label';
  static const String _deviceLabelKey = 'device_label';
  static const String _patientIdKey = 'patient_id';
  static const String _deviceIdKey = 'device_id';
  static const String _authSessionKey = 'auth_session';

  final BackendApiClient _apiClient;
  final SensorStreamingService _sensorService;

  SharedPreferences? _preferences;

  bool _initialized = false;
  bool _isBusy = false;
  bool _backendReachable = false;
  bool _isStreaming = false;

  String _patientName = '';
  int? _patientAge;
  String _roomLabel = '';
  String _deviceLabel = defaultDeviceLabel;

  String? _patientId;
  String? _deviceId;
  String? _sessionId;

  AuthSessionModel? _authSession;

  String _statusMessage = 'Create an account or sign in to begin.';
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
  bool get isAuthenticated => _authSession != null;
  bool get isPatientRole => _authSession?.selectedRole == UserRole.patient;
  bool get isCaregiverRole => _authSession?.selectedRole == UserRole.caregiver;
  bool get canEnableCaregiverRole =>
      isAuthenticated && !availableRoles.contains(UserRole.caregiver);
  bool get hasSetup => _deviceLabel.trim().isNotEmpty;
  bool get isReady => isAuthenticated && hasSetup && _backendReachable;

  AuthSessionModel? get authSession => _authSession;
  AuthUserProfileModel? get currentUser => _authSession?.user;
  UserRole? get selectedRole => _authSession?.selectedRole;
  List<UserRole> get availableRoles =>
      _authSession?.user.availableRoles ?? const <UserRole>[];

  String get patientName =>
      _patientName.trim().isNotEmpty ? _patientName : (_authSession?.user.displayName ?? '');
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
    _patientName = _preferences?.getString(_patientNameKey) ?? '';
    _patientAge = _preferences?.getInt(_patientAgeKey);
    _roomLabel = _preferences?.getString(_roomLabelKey) ?? '';
    _deviceLabel = _preferences?.getString(_deviceLabelKey) ?? defaultDeviceLabel;
    _patientId = _preferences?.getString(_patientIdKey);
    _deviceId = _preferences?.getString(_deviceIdKey);
    _sessionId = null;

    final rawSession = _preferences?.getString(_authSessionKey);
    if (rawSession != null && rawSession.trim().isNotEmpty) {
      try {
        final json = jsonDecode(rawSession);
        if (json is Map<String, dynamic>) {
          _authSession = AuthSessionModel.fromJson(json);
          _apiClient.setAccessToken(_authSession!.accessToken);
          final scopedPatientId = _authSession!.user.patientId;
          if (scopedPatientId != null && scopedPatientId.isNotEmpty) {
            _patientId = scopedPatientId;
          }
        }
      } catch (_) {
        _authSession = null;
        _apiClient.setAccessToken(null);
      }
    }

    if (_authSession != null) {
      await _refreshAuthProfileInternal(updateStatus: false);
    }

    _statusMessage = isAuthenticated
        ? 'Signed in. Select your role-specific workflow from the bottom navigation.'
        : 'Create an account or sign in to begin.';

    _initialized = true;
    notifyListeners();
  }

  Future<void> signUpPatient({
    required String email,
    required String password,
    required String fullName,
    required String patientAgeText,
    required String roomLabel,
  }) async {
    await _ensureInitialized();

    final normalizedFullName = fullName.trim();
    final normalizedEmail = email.trim();
    final normalizedRoomLabel = roomLabel.trim();
    final parsedAge = _parseAgeOrThrow(patientAgeText.trim());

    if (normalizedFullName.isEmpty) {
      _lastError = 'Full name is required.';
      notifyListeners();
      return;
    }
    if (normalizedEmail.isEmpty) {
      _lastError = 'Email is required.';
      notifyListeners();
      return;
    }
    if (password.trim().length < 8) {
      _lastError = 'Password must be at least 8 characters.';
      notifyListeners();
      return;
    }

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Creating patient account...';
    notifyListeners();

    try {
      final session = await _apiClient.signupPatient(
        email: normalizedEmail,
        password: password,
        fullName: normalizedFullName,
        age: parsedAge,
        roomLabel: normalizedRoomLabel.isEmpty ? null : normalizedRoomLabel,
      );

      _patientName = normalizedFullName;
      _patientAge = parsedAge;
      _roomLabel = normalizedRoomLabel;
      _sessionId = null;
      _isStreaming = false;
      _batchesSent = 0;
      _lastBatchSize = 0;
      _lastTransmissionAt = null;
      _lastDetection = null;
      _activeAlert = null;
      _latestTelemetry = null;
      _liveStatus = null;

      await _applyAuthSession(session, persist: true);
      await _persistSetup();
      await _persistIdentifiers();

      _backendReachable = true;
      _statusMessage = 'Account created. You are signed in as Patient.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Signup failed.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> login({
    required String email,
    required String password,
    required UserRole role,
  }) async {
    await _ensureInitialized();

    final normalizedEmail = email.trim();

    if (normalizedEmail.isEmpty) {
      _lastError = 'Email is required.';
      notifyListeners();
      return;
    }
    if (password.trim().length < 8) {
      _lastError = 'Password must be at least 8 characters.';
      notifyListeners();
      return;
    }

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Signing in...';
    notifyListeners();

    try {
      final session = await _apiClient.login(
        email: normalizedEmail,
        password: password,
        role: role,
      );

      _sessionId = null;
      _isStreaming = false;
      await _applyAuthSession(session, persist: true);
      await _persistIdentifiers();

      _backendReachable = true;
      _statusMessage = 'Signed in as ${session.selectedRole.label}.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Login failed.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> switchRole(UserRole role) async {
    await _ensureInitialized();

    if (_authSession == null) {
      _lastError = 'Sign in first.';
      notifyListeners();
      return;
    }

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Switching role to ${role.label}...';
    notifyListeners();

    try {
      if (_isStreaming) {
        await stopMonitoring();
      }

      final session = await _apiClient.switchRole(role);
      await _applyAuthSession(session, persist: true);
      await _persistIdentifiers();

      _statusMessage = 'Role switched to ${role.label}.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Role switch failed.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> enableCaregiverRole() async {
    await _ensureInitialized();

    if (_authSession == null) {
      _lastError = 'Sign in first.';
      notifyListeners();
      return;
    }

    if (availableRoles.contains(UserRole.caregiver)) {
      _statusMessage = 'Caregiver role is already enabled.';
      notifyListeners();
      return;
    }

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Enabling caregiver role...';
    notifyListeners();

    try {
      final profile = await _apiClient.enableCaregiverRole();
      final updatedSession = _authSession!.copyWith(user: profile);
      await _applyAuthSession(updatedSession, persist: true);
      _statusMessage =
          'Caregiver role enabled. Switch roles from the account tab when needed.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Could not enable caregiver role.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> logout() async {
    await _ensureInitialized();

    _isBusy = true;
    _lastError = null;
    _statusMessage = 'Signing out...';
    notifyListeners();

    try {
      if (_isStreaming) {
        await _sensorService.stop();
      }
      _isStreaming = false;
      _sessionId = null;
      _patientId = null;
      _deviceId = null;
      _lastDetection = null;
      _activeAlert = null;
      _latestTelemetry = null;
      _liveStatus = null;
      _batchesSent = 0;
      _lastBatchSize = 0;
      _lastTransmissionAt = null;

      await _clearAuthState(persist: true);
      await _persistIdentifiers();

      _statusMessage = 'Signed out.';
    } catch (error) {
      _lastError = _formatError(error);
      _statusMessage = 'Sign out encountered an issue.';
    } finally {
      _isBusy = false;
      notifyListeners();
    }
  }

  Future<void> refreshAuthProfile({bool silent = false}) async {
    await _ensureInitialized();

    if (!silent) {
      _isBusy = true;
      _lastError = null;
      _statusMessage = 'Refreshing account...';
      notifyListeners();
    }

    try {
      await _refreshAuthProfileInternal(updateStatus: !silent);
    } finally {
      if (!silent) {
        _isBusy = false;
        notifyListeners();
      }
    }
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
    required String patientName,
    required String patientAgeText,
    required String roomLabel,
    required String deviceLabel,
  }) async {
    await _ensureInitialized();

    final normalizedPatientName = patientName.trim();
    final normalizedRoomLabel = roomLabel.trim();
    final normalizedDeviceLabel =
        deviceLabel.trim().isEmpty ? defaultDeviceLabel : deviceLabel.trim();

    int? parsedAge;
    try {
      parsedAge = _parseAgeOrThrow(patientAgeText.trim());
    } catch (error) {
      _lastError = _formatError(error);
      notifyListeners();
      return;
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
      _patientName = normalizedPatientName;
      _patientAge = parsedAge;
      _roomLabel = normalizedRoomLabel;
      _deviceLabel = normalizedDeviceLabel;

      if (patientChanged && !isAuthenticated) {
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

    if (!isAuthenticated) {
      _lastError = 'Sign in before starting live monitoring.';
      _statusMessage = 'Authentication required.';
      notifyListeners();
      return;
    }

    if (!isPatientRole) {
      _lastError = 'Live sensor streaming is available only in Patient mode.';
      _statusMessage = 'Switch to Patient role to start monitoring.';
      notifyListeners();
      return;
    }

    if (!hasSetup) {
      _lastError = 'Complete the device label first.';
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
          'The backend is not reachable. Verify the server is running and reachable from this device network.',
        );
      }

      final sensorStatus = await refreshSensorStatus(silent: true);
      if (!sensorStatus.allAvailable) {
        throw ApiException(
          'Required phone sensors are not available. Check accelerometer and gyroscope access.',
        );
      }

      await _ensurePatient(allowCreate: true);
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
        patientName: patientName,
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

    if (!isAuthenticated) {
      _lastError = 'Sign in before triggering an emergency alert.';
      _statusMessage = 'Authentication required.';
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

      await _ensurePatient(allowCreate: isPatientRole);

      if (isPatientRole) {
        await _ensureDevice();
      }

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

  Future<void> _ensurePatient({required bool allowCreate}) async {
    final scopedPatientId = _authSession?.user.patientId;
    if (scopedPatientId != null && scopedPatientId.isNotEmpty) {
      _patientId = scopedPatientId;
    }

    if (_patientId != null) {
      try {
        final patient = await _apiClient.getPatient(_patientId!);
        _patientName = patient.fullName;
        _patientAge = patient.age ?? _patientAge;
        _roomLabel = patient.roomLabel ?? _roomLabel;
        await _persistSetup();
        await _persistIdentifiers();
        return;
      } catch (_) {
        _patientId = null;
      }
    }

    if (!allowCreate) {
      throw ApiException('No patient profile is linked to this account role.');
    }

    final fallbackName = patientName.trim().isEmpty ? 'Patient User' : patientName;
    final patient = await _apiClient.createPatient(
      fullName: fallbackName,
      age: _patientAge,
      roomLabel: _roomLabel.isEmpty ? null : _roomLabel,
    );

    _patientId = patient.id;
    _patientName = patient.fullName;
    _patientAge = patient.age ?? _patientAge;
    _roomLabel = patient.roomLabel ?? _roomLabel;
    await _persistSetup();
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
      ownerName: patientName,
    );

    _deviceId = device.id;
    await _persistIdentifiers();
  }

  Future<bool> _refreshAuthProfileInternal({required bool updateStatus}) async {
    if (_authSession == null) {
      return false;
    }

    try {
      final profile = await _apiClient.fetchProfile();
      AuthSessionModel updatedSession = _authSession!.copyWith(user: profile);

      if (!profile.availableRoles.contains(updatedSession.selectedRole) &&
          profile.availableRoles.isNotEmpty) {
        updatedSession = await _apiClient.switchRole(profile.availableRoles.first);
      }

      await _applyAuthSession(updatedSession, persist: true);
      await _persistIdentifiers();
      if (updateStatus) {
        _statusMessage = 'Signed in as ${updatedSession.selectedRole.label}.';
      }
      return true;
    } catch (error) {
      await _clearAuthState(persist: true);
      _patientId = null;
      await _persistIdentifiers();

      if (updateStatus) {
        _lastError = _formatError(error);
        _statusMessage = 'Your session expired. Please sign in again.';
      }
      return false;
    }
  }

  Future<void> _applyAuthSession(AuthSessionModel session, {required bool persist}) async {
    _authSession = session;
    _apiClient.setAccessToken(session.accessToken);

    final scopedPatientId = session.user.patientId;
    if (scopedPatientId != null && scopedPatientId.isNotEmpty) {
      _patientId = scopedPatientId;
    }

    if (_patientName.trim().isEmpty) {
      _patientName = session.user.displayName;
    }

    if (persist) {
      await _persistAuthSession();
    }
  }

  Future<void> _clearAuthState({required bool persist}) async {
    _authSession = null;
    _apiClient.setAccessToken(null);
    if (persist) {
      final preferences = _preferences ?? await SharedPreferences.getInstance();
      _preferences = preferences;
      await preferences.remove(_authSessionKey);
    }
  }

  int? _parseAgeOrThrow(String rawAge) {
    if (rawAge.isEmpty) {
      return null;
    }

    final parsed = int.tryParse(rawAge);
    if (parsed == null || parsed < 0 || parsed > 130) {
      throw ApiException('Patient age must be a whole number between 0 and 130.');
    }
    return parsed;
  }

  Future<void> _persistSetup() async {
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;

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

  Future<void> _persistAuthSession() async {
    final preferences = _preferences ?? await SharedPreferences.getInstance();
    _preferences = preferences;

    if (_authSession == null) {
      await preferences.remove(_authSessionKey);
      return;
    }

    await preferences.setString(_authSessionKey, jsonEncode(_authSession!.toJson()));
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
