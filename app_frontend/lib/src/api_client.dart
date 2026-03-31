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

  Future<Map<String, dynamic>> _send(Future<http.Response> requestFuture) async {
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

  Future<Map<String, dynamic>> _decodeResponse(http.Response response) async {
    final body = response.body.trim();
    final jsonBody = body.isEmpty ? <String, dynamic>{} : jsonDecode(body);

    if (response.statusCode < 200 || response.statusCode >= 300) {
      if (jsonBody is Map<String, dynamic>) {
        throw ApiException(jsonBody['detail']?.toString() ?? 'Request failed (${response.statusCode})');
      }
      throw ApiException('Request failed (${response.statusCode})');
    }

    if (jsonBody is Map<String, dynamic>) {
      return jsonBody;
    }
    throw ApiException('Unexpected response format from backend.');
  }

  Future<void> ping() async {
    await _send(http.get(_uri('/api/v1/health')));
  }

  Future<PatientRecord> createPatient({
    required String fullName,
    int? age,
    String? roomLabel,
  }) async {
    return PatientRecord.fromJson(
      await _send(
        http.post(
          _uri('/api/v1/patients'),
          headers: {'Content-Type': 'application/json'},
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
      await _send(http.get(_uri('/api/v1/patients/$patientId'))),
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
          _uri('/api/v1/devices'),
          headers: {'Content-Type': 'application/json'},
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
      await _send(http.get(_uri('/api/v1/devices/$deviceId'))),
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
          _uri('/api/v1/sessions'),
          headers: {'Content-Type': 'application/json'},
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
      await _send(
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
      ),
    );
  }
}
