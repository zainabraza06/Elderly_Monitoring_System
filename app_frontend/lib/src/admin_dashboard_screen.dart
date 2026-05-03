import 'package:flutter/material.dart';

import 'api_client.dart';
import 'monitoring_controller.dart' show MonitoringController;

/// Administrator metrics (`GET /api/v1/admin/dashboard`).
class AdminDashboardScreen extends StatefulWidget {
  const AdminDashboardScreen({super.key, required this.onBack});

  final VoidCallback onBack;

  @override
  State<AdminDashboardScreen> createState() => _AdminDashboardScreenState();
}

class _AdminDashboardScreenState extends State<AdminDashboardScreen> {
  final _emailCtrl = TextEditingController(text: 'admin@local');
  final _passCtrl = TextEditingController(text: 'admin123');
  final _urlCtrl = TextEditingController(text: MonitoringController.defaultBackendUrl);

  Map<String, dynamic>? _dash;
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    _emailCtrl.dispose();
    _passCtrl.dispose();
    _urlCtrl.dispose();
    super.dispose();
  }

  Future<void> _loginAndLoad() async {
    setState(() {
      _busy = true;
      _error = null;
      _dash = null;
    });
    try {
      final c = BackendApiClient(baseUrl: _urlCtrl.text.trim());
      final login = await c.adminLogin(email: _emailCtrl.text.trim(), password: _passCtrl.text);
      final tok = login['access_token'] as String? ?? '';
      c.setBearerToken(tok);
      final d = await c.getAdminDashboard();
      setState(() => _dash = d);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(icon: const Icon(Icons.arrow_back), onPressed: widget.onBack),
        title: const Text('Admin — SisFall'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          TextField(
            controller: _urlCtrl,
            decoration: const InputDecoration(labelText: 'Backend base URL'),
          ),
          const SizedBox(height: 10),
          TextField(
            controller: _emailCtrl,
            decoration: const InputDecoration(labelText: 'Admin email'),
          ),
          const SizedBox(height: 10),
          TextField(
            controller: _passCtrl,
            obscureText: true,
            decoration: const InputDecoration(labelText: 'Password'),
          ),
          const SizedBox(height: 14),
          if (_error != null)
            Text(_error!, style: const TextStyle(color: Color(0xFFB53B34))),
          FilledButton(
            onPressed: _busy ? null : _loginAndLoad,
            child: Text(_busy ? 'Loading…' : 'Sign in & refresh metrics'),
          ),
          const SizedBox(height: 24),
          if (_dash != null) ...[
            const Text('Overview', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18)),
            const SizedBox(height: 8),
            _tile('Caretakers', '${_dash!['caretakers']}'),
            _tile('Elders (accounts)', '${_dash!['elders_registered']}'),
            _tile('Patients', '${_dash!['patients']}'),
            _tile('Open alerts', '${_dash!['open_alerts']}'),
            _tile('Fall feedback rows', '${_dash!['fall_feedback_events']}'),
            _tile('Datasets', '${_dash!['datasets']}'),
            if (_dash!['note'] != null) Text('${_dash!['note']}'),
          ],
        ],
      ),
    );
  }

  Widget _tile(String k, String v) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Row(
          children: [
            Expanded(child: Text(k)),
            Text(v, style: const TextStyle(fontWeight: FontWeight.w600)),
          ],
        ),
      );
}
