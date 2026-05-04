class AppApiConfig {
  AppApiConfig._();

  // Production backend (Render). Override in-app via setup if you need a local server.
  static const String backendBaseUrl = 'https://safestep-ai.onrender.com';

  // Shared API prefix used for all backend REST calls.
  static const String apiPrefix = '/api/v1';
}
