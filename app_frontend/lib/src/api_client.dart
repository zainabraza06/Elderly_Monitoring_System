import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'models.dart';


class BackendApiClient {
  BackendApiClient({required String baseUrl}) : _baseUrl = _normalizeBaseUrl(baseUrl);

  static const Duration _requestTimeout = Duration(seconds: 8);

  String _baseUrl;

  static String _normalizeBaseUrl(String input) {
    final trimmed = input.trim();
    final withScheme = trimmed.startsWith('http://') || trimmed.startsWith('https://')
        ? trimmed
        : 'http://$trimmed';
    if (withScheme.endsWith('/')) {
      return withScheme.substring(0, withScheme.length - 1);
    }
    return withScheme;
  }

  void updateBaseUrl(String baseUrl) {
    _baseUrl = _normalizeBaseUrl(baseUrl);
  }

  Uri _uri(String path) => Uri.parse('$_baseUrl$path');

  Future<Object> _send(Future<http.Response> requestFuture) async {
    try {
      final response = await requestFuture.timeout(_requestTimeout);
      return await _decodeResponse(response);
    } on TimeoutException {
      throw ApiException(
        'The backend request timed out. Check the server status and the base URL.',
      );
    } catch (error) {
      if (error is ApiException) {
        rethrow;
      }
      throw ApiException('Could not reach the backend: $error');
    }
  }

  Future<Object> _decodeResponse(http.Response response) async {
    final body = response.body.trim();
    final jsonBody = body.isEmpty ? <String, dynamic>{} : jsonDecode(body);

    if (response.statusCode < 200 || response.statusCode >= 300) {
      if (jsonBody is Map<String, dynamic>) {
        throw ApiException(jsonBody['detail']?.toString() ?? 'Request failed (${response.statusCode})');
      }
      throw ApiException('Request failed (${response.statusCode})');
    }

    if (jsonBody is Map<String, dynamic> || jsonBody is List<dynamic>) {
      return jsonBody as Object;
    }
    throw ApiException('Unexpected response format from backend.');
  }

  Map<String, dynamic> _asMap(Object payload) {
    if (payload is Map<String, dynamic>) {
      return payload;
    }
    throw ApiException('Unexpected response object format from backend.');
  }

  List<dynamic> _asList(Object payload) {
    if (payload is List<dynamic>) {
      return payload;
    }
    throw ApiException('Unexpected response list format from backend.');
  }

  /// Returns manifest-backed dimensions when the inference stack is loaded (503 if not).
  Future<Map<String, dynamic>> getInferenceStatus() async {
    return _asMap(
      await _send(http.get(_uri('/api/v1/inference/status'))),
    );
  }

  Future<void> ping() async {
    await _send(http.get(_uri('/api/v1/health')));
  }

  Future<PatientRecord> createPatient({
    required String fullName,
    int? age,
  }) async {
    return PatientRecord.fromJson(
      _asMap(await _send(
        http.post(
          _uri('/api/v1/patients'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'full_name': fullName,
            'age': age,
          }),
        ),
      )),
    );
  }

  Future<PatientRecord> getPatient(String patientId) async {
    return PatientRecord.fromJson(
      _asMap(await _send(http.get(_uri('/api/v1/patients/$patientId')))),
    );
  }

  Future<DeviceRecord> createDevice({
    required String label,
    String platform = 'flutter_mobile',
    String? ownerName,
  }) async {
    return DeviceRecord.fromJson(
      _asMap(await _send(
        http.post(
          _uri('/api/v1/devices'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'label': label,
            'platform': platform,
            'owner_name': ownerName,
          }),
        ),
      )),
    );
  }

  Future<DeviceRecord> getDevice(String deviceId) async {
    return DeviceRecord.fromJson(
      _asMap(await _send(http.get(_uri('/api/v1/devices/$deviceId')))),
    );
  }

  Future<SessionRecord> startSession({
    required String patientId,
    required String deviceId,
    required double sampleRateHz,
    String startedBy = 'flutter_app',
  }) async {
    return SessionRecord.fromJson(
      _asMap(await _send(
        http.post(
          _uri('/api/v1/sessions'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'patient_id': patientId,
            'device_id': deviceId,
            'sample_rate_hz': sampleRateHz,
            'started_by': startedBy,
          }),
        ),
      )),
    );
  }

  Future<void> stopSession(String sessionId) async {
    await _send(
      http.post(
        _uri('/api/v1/sessions/$sessionId/stop'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'stopped_by': 'flutter_app',
          'note': 'Stopped from mobile app.',
        }),
      ),
    );
  }

  Future<IngestResponseModel> ingestLiveBatch({
    required String patientId,
    required String deviceId,
    required String sessionId,
    required double samplingRateHz,
    required double? batteryLevel,
    required List<SensorReadingPayload> samples,
  }) async {
    return IngestResponseModel.fromJson(
      _asMap(await _send(
        http.post(
          _uri('/api/v1/ingest/live'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'patient_id': patientId,
            'device_id': deviceId,
            'session_id': sessionId,
            'source': 'flutter_mobile',
            'sampling_rate_hz': samplingRateHz,
            'acceleration_unit': 'm_s2',
            'gyroscope_unit': 'rad_s',
            'battery_level': batteryLevel,
            'samples': samples.map((sample) => sample.toJson()).toList(),
          }),
        ),
      )),
    );
  }

  Future<AlertRecordModel> triggerManualAlert({
    required String patientId,
    required String? deviceId,
    required String? sessionId,
    String severity = 'fall_detected',
    String message = 'Emergency alert triggered from mobile app.',
  }) async {
    return AlertRecordModel.fromJson(
      _asMap(await _send(
        http.post(
          _uri('/api/v1/alerts/manual'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'patient_id': patientId,
            'device_id': deviceId,
            'session_id': sessionId,
            'severity': severity,
            'message': message,
            'actor': 'flutter_app',
          }),
        ),
      )),
    );
  }

  Future<SystemSummaryModel> getSummary() async {
    return SystemSummaryModel.fromJson(
      _asMap(await _send(http.get(_uri('/api/v1/summary')))),
    );
  }

  Future<List<LiveStatusModel>> getLivePatients() async {
    final rows = _asList(await _send(http.get(_uri('/api/v1/monitor/patients/live'))));
    return rows
        .whereType<Map<String, dynamic>>()
        .map(LiveStatusModel.fromJson)
        .toList();
  }

  Future<List<AlertRecordModel>> getAlerts({
    String? status,
    String? patientId,
  }) async {
    final query = <String, String>{};
    if (status != null && status.isNotEmpty) {
      query['status'] = status;
    }
    if (patientId != null && patientId.isNotEmpty) {
      query['patient_id'] = patientId;
    }

    final uri = _uri('/api/v1/alerts').replace(queryParameters: query.isEmpty ? null : query);
    final rows = _asList(await _send(http.get(uri)));
    return rows
        .whereType<Map<String, dynamic>>()
        .map(AlertRecordModel.fromJson)
        .toList();
  }

  Future<AlertRecordModel> acknowledgeAlert({
    required String alertId,
    String actor = 'caregiver_app',
    String? note,
  }) async {
    return AlertRecordModel.fromJson(
      _asMap(await _send(
        http.post(
          _uri('/api/v1/alerts/$alertId/acknowledge'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'actor': actor,
            'note': note,
          }),
        ),
      )),
    );
  }

  Future<AlertRecordModel> resolveAlert({
    required String alertId,
    String actor = 'caregiver_app',
    String? note,
  }) async {
    return AlertRecordModel.fromJson(
      _asMap(await _send(
        http.post(
          _uri('/api/v1/alerts/$alertId/resolve'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'actor': actor,
            'note': note,
          }),
        ),
      )),
    );
  }

  Future<void> updateDetectorSensitivity(String level) async {
    Map<String, dynamic> payload;
    switch (level) {
      case 'low':
        payload = {
          'medium_risk_score': 0.45,
          'high_risk_score': 0.68,
          'fall_score': 0.88,
        };
      case 'high':
        payload = {
          'medium_risk_score': 0.28,
          'high_risk_score': 0.50,
          'fall_score': 0.72,
        };
      default:
        payload = {
          'medium_risk_score': 0.35,
          'high_risk_score': 0.58,
          'fall_score': 0.80,
        };
    }

    await _send(
      http.put(
        _uri('/api/v1/detector/config'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      ),
    );
  }

  Future<CaregiverAuthModel> caregiverSignup({
    required String fullName,
    required String email,
    required String password,
  }) async {
    return CaregiverAuthModel.fromJson(
      _asMap(await _send(
        http.post(
          _uri('/api/v1/auth/caregiver/signup'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'full_name': fullName,
            'email': email,
            'password': password,
          }),
        ),
      )),
    );
  }

  Future<CaregiverAuthModel> caregiverLogin({
    required String email,
    required String password,
  }) async {
    return CaregiverAuthModel.fromJson(
      _asMap(await _send(
        http.post(
          _uri('/api/v1/auth/caregiver/login'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'email': email,
            'password': password,
          }),
        ),
      )),
    );
  }

  Future<GeneratedPatientCredentialModel> generatePatientCredentials({
    required String caregiverToken,
    required String fullName,
    required int? age,
    required String homeAddress,
    String? emergencyContact,
    String? notes,
  }) async {
    return GeneratedPatientCredentialModel.fromJson(
      _asMap(await _send(
        http.post(
          _uri('/api/v1/auth/caregiver/patient-credentials'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'caregiver_token': caregiverToken,
            'full_name': fullName,
            'age': age,
            'home_address': homeAddress,
            'emergency_contact': emergencyContact,
            'notes': notes,
          }),
        ),
      )),
    );
  }

  /// XGBoost pipeline: `enhancedFeatures` length = `enhanced_feature_dim` in `models/inference_manifest.json`.
  /// When fall is predicted, send `fallTypeFeatures` with length `fall_type_raw_dim` (263 for Colab fall-type).
  Future<Map<String, dynamic>> inferMotion({
    required List<double> enhancedFeatures,
    List<double>? fallTypeFeatures,
    bool predictFallType = true,
  }) async {
    return _asMap(
      await _send(
        http.post(
          _uri('/api/v1/inference/motion'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'enhanced_features': enhancedFeatures,
            if (fallTypeFeatures != null) 'fall_type_features': fallTypeFeatures,
            'predict_fall_type': predictFallType,
          }),
        ),
      ),
    );
  }
}
