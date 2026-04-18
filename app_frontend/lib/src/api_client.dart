import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'api_config.dart';
import 'models.dart';

class BackendApiClient {
  BackendApiClient({String? baseUrl})
    : _baseUrl = _normalizeBaseUrl(baseUrl ?? AppApiConfig.backendBaseUrl);

  static const Duration _requestTimeout = Duration(seconds: 8);

  String _baseUrl;
  String? _accessToken;

  static String _normalizeBaseUrl(String input) {
    final trimmed = input.trim();
    final withScheme =
        trimmed.startsWith('http://') || trimmed.startsWith('https://')
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

  void setAccessToken(String? token) {
    final normalized = token?.trim();
    _accessToken = normalized == null || normalized.isEmpty ? null : normalized;
  }

  Map<String, String> _headers({
    bool includeJsonContentType = true,
    bool includeAuthorization = true,
  }) {
    final headers = <String, String>{};

    if (includeJsonContentType) {
      headers['Content-Type'] = 'application/json';
    }

    if (includeAuthorization && _accessToken != null) {
      headers['Authorization'] = 'Bearer $_accessToken';
    }

    return headers;
  }

  Uri _uri(String path) {
    final normalizedPath = path.startsWith('/') ? path : '/$path';
    return Uri.parse('$_baseUrl${AppApiConfig.apiPrefix}$normalizedPath');
  }

  Future<Map<String, dynamic>> _send(
    Future<http.Response> requestFuture,
  ) async {
    try {
      final response = await requestFuture.timeout(_requestTimeout);
      return await _decodeResponse(response);
    } on TimeoutException {
      throw ApiException(
        'The backend request timed out. Check server status and network connectivity.',
      );
    } catch (error) {
      if (error is ApiException) {
        rethrow;
      }
      throw ApiException('Could not reach the backend: $error');
    }
  }

  Future<Map<String, dynamic>> _decodeResponse(http.Response response) async {
    final body = response.body.trim();
    final jsonBody = body.isEmpty ? <String, dynamic>{} : jsonDecode(body);

    if (response.statusCode < 200 || response.statusCode >= 300) {
      if (jsonBody is Map<String, dynamic>) {
        final message =
            jsonBody['detail']?.toString() ??
            jsonBody['message']?.toString() ??
            'Request failed (${response.statusCode})';
        throw ApiException(message);
      }
      throw ApiException('Request failed (${response.statusCode})');
    }

    if (jsonBody is Map<String, dynamic>) {
      return jsonBody;
    }
    throw ApiException('Unexpected response format from backend.');
  }

  Future<void> ping() async {
    await _send(http.get(_uri('/health')));
  }

  Future<AuthSessionModel> signupPatient({
    required String email,
    required String password,
    required String fullName,
    int? age,
    String? roomLabel,
  }) async {
    return AuthSessionModel.fromJson(
      await _send(
        http.post(
          _uri('/auth/signup/patient'),
          headers: _headers(includeAuthorization: false),
          body: jsonEncode({
            'email': email,
            'password': password,
            'full_name': fullName,
            'age': age,
            'room_label': roomLabel,
          }),
        ),
      ),
    );
  }

  Future<AuthSessionModel> login({
    required String email,
    required String password,
    required UserRole role,
  }) async {
    return AuthSessionModel.fromJson(
      await _send(
        http.post(
          _uri('/auth/login'),
          headers: _headers(includeAuthorization: false),
          body: jsonEncode({
            'email': email,
            'password': password,
            'role': role.value,
          }),
        ),
      ),
    );
  }

  Future<AuthUserProfileModel> fetchProfile() async {
    return AuthUserProfileModel.fromJson(
      await _send(
        http.get(
          _uri('/auth/me'),
          headers: _headers(includeJsonContentType: false),
        ),
      ),
    );
  }

  Future<AuthUserProfileModel> enableCaregiverRole() async {
    return AuthUserProfileModel.fromJson(
      await _send(
        http.post(_uri('/auth/roles/caregiver'), headers: _headers()),
      ),
    );
  }

  Future<AuthSessionModel> switchRole(UserRole role) async {
    return AuthSessionModel.fromJson(
      await _send(
        http.post(
          _uri('/auth/switch-role'),
          headers: _headers(),
          body: jsonEncode({'role': role.value}),
        ),
      ),
    );
  }

  Future<PatientRecord> createPatient({
    required String fullName,
    int? age,
    String? roomLabel,
  }) async {
    return PatientRecord.fromJson(
      await _send(
        http.post(
          _uri('/patients'),
          headers: _headers(),
          body: jsonEncode({
            'full_name': fullName,
            'age': age,
            'room_label': roomLabel,
          }),
        ),
      ),
    );
  }

  Future<PatientRecord> getPatient(String patientId) async {
    return PatientRecord.fromJson(
      await _send(
        http.get(
          _uri('/patients/$patientId'),
          headers: _headers(includeJsonContentType: false),
        ),
      ),
    );
  }

  Future<DeviceRecord> createDevice({
    required String label,
    String platform = 'flutter_mobile',
    String? ownerName,
  }) async {
    return DeviceRecord.fromJson(
      await _send(
        http.post(
          _uri('/devices'),
          headers: _headers(),
          body: jsonEncode({
            'label': label,
            'platform': platform,
            'owner_name': ownerName,
          }),
        ),
      ),
    );
  }

  Future<DeviceRecord> getDevice(String deviceId) async {
    return DeviceRecord.fromJson(
      await _send(
        http.get(
          _uri('/devices/$deviceId'),
          headers: _headers(includeJsonContentType: false),
        ),
      ),
    );
  }

  Future<SessionRecord> startSession({
    required String patientId,
    required String deviceId,
    required double sampleRateHz,
    String startedBy = 'flutter_app',
  }) async {
    return SessionRecord.fromJson(
      await _send(
        http.post(
          _uri('/sessions'),
          headers: _headers(),
          body: jsonEncode({
            'patient_id': patientId,
            'device_id': deviceId,
            'sample_rate_hz': sampleRateHz,
            'started_by': startedBy,
          }),
        ),
      ),
    );
  }

  Future<void> stopSession(String sessionId) async {
    await _send(
      http.post(
        _uri('/sessions/$sessionId/stop'),
        headers: _headers(),
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
      await _send(
        http.post(
          _uri('/ingest/live'),
          headers: _headers(),
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
      ),
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
      await _send(
        http.post(
          _uri('/alerts/manual'),
          headers: _headers(),
          body: jsonEncode({
            'patient_id': patientId,
            'device_id': deviceId,
            'session_id': sessionId,
            'severity': severity,
            'message': message,
            'actor': 'flutter_app',
          }),
        ),
      ),
    );
  }
}
