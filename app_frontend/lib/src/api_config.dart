class AppApiConfig {
  AppApiConfig._();

  // Change this once to point the app to a different backend host.
  static const String backendBaseUrl = 'http://10.0.2.2:8000';

  // Shared API prefix used for all backend REST calls.
  static const String apiPrefix = '/api/v1';
}