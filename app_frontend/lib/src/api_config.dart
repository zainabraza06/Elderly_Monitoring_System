class AppApiConfig {
  AppApiConfig._();

  // Change this once to point the app to a different backend host.
  static const String backendBaseUrl =
      'https://detection-backend.app.zeeshan-abbas.tech/';

  // Shared API prefix used for all backend REST calls.
  static const String apiPrefix = '/api/v1';
}
