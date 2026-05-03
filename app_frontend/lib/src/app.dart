import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import 'models.dart';
import 'monitoring_controller.dart';

class ElderlyMonitorApp extends StatefulWidget {
  const ElderlyMonitorApp({super.key, this.controller});

  final MonitoringController? controller;

  @override
  State<ElderlyMonitorApp> createState() => _ElderlyMonitorAppState();
}

class _ElderlyMonitorAppState extends State<ElderlyMonitorApp> {
  late final MonitoringController _controller;
  late final Future<void> _initializationFuture;
  late final bool _ownsController;

  @override
  void initState() {
    super.initState();
    _ownsController = widget.controller == null;
    _controller = widget.controller ?? MonitoringController();
    _initializationFuture = _controller.initialize();
  }

  @override
  void dispose() {
    if (_ownsController) {
      _controller.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Care Monitor',
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF5F7FA),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0A7FA6),
          brightness: Brightness.light,
        ),
      ),
      home: FutureBuilder<void>(
        future: _initializationFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const Scaffold(body: Center(child: CircularProgressIndicator()));
          }
          return MonitoringShell(controller: _controller);
        },
      ),
    );
  }
}

class MonitoringShell extends StatefulWidget {
  const MonitoringShell({super.key, required this.controller});

  final MonitoringController controller;

  @override
  State<MonitoringShell> createState() => _MonitoringShellState();
}

class _MonitoringShellState extends State<MonitoringShell> {
  int _tabIndex = 0;
  bool _patientMode = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      widget.controller.refreshCaregiverData(silent: true);
    });
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.controller,
      builder: (context, _) {
        final controller = widget.controller;
        if (!controller.isCaregiverAuthenticated) {
          return CaregiverAuthScreen(controller: controller);
        }
        final body = _patientMode
            ? PatientModeHome(controller: controller)
            : _buildCaregiverTabs(controller);

        return Scaffold(
          appBar: AppBar(
            title: Text(_patientMode ? 'Patient Mode' : 'Caregiver Console'),
            actions: [
              Padding(
                padding: const EdgeInsets.only(right: 12),
                child: SegmentedButton<bool>(
                  segments: const [
                    ButtonSegment<bool>(value: false, label: Text('Caregiver')),
                    ButtonSegment<bool>(value: true, label: Text('Patient')),
                  ],
                  selected: {_patientMode},
                  onSelectionChanged: (value) {
                    setState(() => _patientMode = value.first);
                  },
                ),
              ),
            ],
          ),
          body: Stack(
            children: [
              Column(
                children: [
                  if (controller.isAlarmPlaying)
                    Container(
                      width: double.infinity,
                      color: const Color(0xFFB53B34),
                      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                      child: SafeArea(
                        bottom: false,
                        child: Row(
                          children: [
                            const Icon(Icons.notification_important, color: Colors.white),
                            const SizedBox(width: 8),
                            const Expanded(
                              child: Text(
                                'Emergency alarm is active',
                                style: TextStyle(color: Colors.white, fontWeight: FontWeight.w700),
                              ),
                            ),
                            TextButton(
                              onPressed: controller.clearActiveAlarm,
                              style: TextButton.styleFrom(foregroundColor: Colors.white),
                              child: const Text('Clear'),
                            ),
                          ],
                        ),
                      ),
                    ),
                  Expanded(child: body),
                ],
              ),
              if (controller.isBusy)
                const Align(
                  alignment: Alignment.topCenter,
                  child: LinearProgressIndicator(minHeight: 3),
                ),
            ],
          ),
          bottomNavigationBar: _patientMode
              ? null
              : NavigationBar(
                  selectedIndex: _tabIndex,
                  onDestinationSelected: (value) => setState(() => _tabIndex = value),
                  destinations: const [
                    NavigationDestination(icon: Icon(Icons.home_outlined), label: 'Home'),
                    NavigationDestination(icon: Icon(Icons.badge_outlined), label: 'Enrollment'),
                    NavigationDestination(icon: Icon(Icons.monitor_heart_outlined), label: 'Live'),
                    NavigationDestination(icon: Icon(Icons.notification_important_outlined), label: 'Alerts'),
                    NavigationDestination(icon: Icon(Icons.insights_outlined), label: 'Insights'),
                    NavigationDestination(icon: Icon(Icons.person_outline), label: 'Profile'),
                    NavigationDestination(icon: Icon(Icons.settings_outlined), label: 'Settings'),
                  ],
                ),
        );
      },
    );
  }

  Widget _buildCaregiverTabs(MonitoringController controller) {
    final tabs = <Widget>[
      CaregiverDashboard(controller: controller, openAlerts: () => setState(() => _tabIndex = 3)),
      CaregiverEnrollmentScreen(controller: controller),
      LiveMonitoringScreen(controller: controller),
      AlertsScreen(controller: controller),
      InsightsScreen(controller: controller),
      PatientProfileScreen(controller: controller),
      SettingsScreen(controller: controller),
    ];
    return tabs[_tabIndex];
  }
}

class CaregiverAuthScreen extends StatefulWidget {
  const CaregiverAuthScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  State<CaregiverAuthScreen> createState() => _CaregiverAuthScreenState();
}

class _CaregiverAuthScreenState extends State<CaregiverAuthScreen> {
  bool _isSignup = true;
  final _nameCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _passwordCtrl = TextEditingController();

  @override
  void dispose() {
    _nameCtrl.dispose();
    _emailCtrl.dispose();
    _passwordCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (_isSignup) {
      await widget.controller.caregiverSignup(
        fullName: _nameCtrl.text,
        email: _emailCtrl.text,
        password: _passwordCtrl.text,
      );
    } else {
      await widget.controller.caregiverLogin(
        email: _emailCtrl.text,
        password: _passwordCtrl.text,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    return Scaffold(
      appBar: AppBar(title: const Text('Caregiver Access')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Container(
            padding: const EdgeInsets.all(18),
            decoration: _cardDecoration(),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _isSignup ? 'Create caregiver account' : 'Caregiver sign in',
                  style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 12),
                if (_isSignup) ...[
                  TextField(
                    controller: _nameCtrl,
                    decoration: const InputDecoration(labelText: 'Full Name'),
                  ),
                  const SizedBox(height: 10),
                ],
                TextField(
                  controller: _emailCtrl,
                  keyboardType: TextInputType.emailAddress,
                  decoration: const InputDecoration(labelText: 'Email'),
                ),
                const SizedBox(height: 10),
                TextField(
                  controller: _passwordCtrl,
                  obscureText: true,
                  decoration: const InputDecoration(labelText: 'Password'),
                ),
                const SizedBox(height: 14),
                FilledButton(
                  onPressed: controller.isBusy ? null : _submit,
                  child: Text(_isSignup ? 'Sign Up as Caregiver' : 'Sign In'),
                ),
                const SizedBox(height: 8),
                TextButton(
                  onPressed: () => setState(() => _isSignup = !_isSignup),
                  child: Text(
                    _isSignup ? 'Already have an account? Sign in' : 'Need an account? Sign up',
                  ),
                ),
              ],
            ),
          ),
          if (controller.lastError != null) ...[
            const SizedBox(height: 12),
            _StatusBanner(
              color: const Color(0xFFB53B34),
              title: 'Sign-in Error',
              message: controller.lastError!,
            ),
          ],
        ],
      ),
    );
  }
}

class CaregiverEnrollmentScreen extends StatefulWidget {
  const CaregiverEnrollmentScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  State<CaregiverEnrollmentScreen> createState() => _CaregiverEnrollmentScreenState();
}

class _CaregiverEnrollmentScreenState extends State<CaregiverEnrollmentScreen> {
  final _nameCtrl = TextEditingController();
  final _ageCtrl = TextEditingController();
  final _homeCtrl = TextEditingController();
  final _contactCtrl = TextEditingController();
  final _notesCtrl = TextEditingController();

  @override
  void dispose() {
    _nameCtrl.dispose();
    _ageCtrl.dispose();
    _homeCtrl.dispose();
    _contactCtrl.dispose();
    _notesCtrl.dispose();
    super.dispose();
  }

  Future<void> _generate() async {
    await widget.controller.generatePatientCredentials(
      fullName: _nameCtrl.text,
      ageText: _ageCtrl.text,
      homeAddress: _homeCtrl.text,
      emergencyContact: _contactCtrl.text,
      notes: _notesCtrl.text,
    );
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    final generated = controller.lastGeneratedCredential;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _StatusBanner(
          color: const Color(0xFF2A7DA8),
          title: 'Patient Enrollment',
          message: 'Create patient credentials securely for patient app access.',
        ),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: _cardDecoration(),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              TextField(controller: _nameCtrl, decoration: const InputDecoration(labelText: 'Patient Name')),
              const SizedBox(height: 8),
              TextField(
                controller: _ageCtrl,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(labelText: 'Age'),
              ),
              const SizedBox(height: 8),
              TextField(controller: _homeCtrl, decoration: const InputDecoration(labelText: 'Home Address')),
              const SizedBox(height: 8),
              TextField(
                controller: _contactCtrl,
                decoration: const InputDecoration(labelText: 'Emergency Contact'),
              ),
              const SizedBox(height: 8),
              TextField(controller: _notesCtrl, decoration: const InputDecoration(labelText: 'Notes')),
              const SizedBox(height: 12),
              FilledButton.icon(
                onPressed: controller.isBusy ? null : _generate,
                icon: const Icon(Icons.vpn_key_outlined),
                label: const Text('Generate Credentials for Patient'),
              ),
            ],
          ),
        ),
        if (generated != null) ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: _cardDecoration(),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Generated Patient Access', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18)),
                const SizedBox(height: 8),
                _StatCard(label: 'Patient', value: generated.patientName),
                const SizedBox(height: 8),
                _StatCard(label: 'Username', value: generated.username),
                const SizedBox(height: 8),
                _StatCard(label: 'Temporary Password', value: generated.temporaryPassword),
                const SizedBox(height: 8),
                const Text(
                  'Share these credentials securely with the patient.',
                  style: TextStyle(color: Color(0xFF5D7385)),
                ),
              ],
            ),
          ),
        ],
      ],
    );
  }
}

class CaregiverDashboard extends StatelessWidget {
  const CaregiverDashboard({super.key, required this.controller, required this.openAlerts});

  final MonitoringController controller;
  final VoidCallback openAlerts;

  @override
  Widget build(BuildContext context) {
    final live = controller.liveStatus;
    final severity = live?.severity ?? 'low';
    final risk = ((live?.score ?? controller.lastDetection?.score ?? 0) * 100).round();
    final statusText = live?.lastMessage ?? 'Patient is stable.';
    final movement = (controller.lastDetection?.fallProbability ?? live?.fallProbability ?? 0) * 100;

    return RefreshIndicator(
      onRefresh: () => controller.refreshCaregiverData(),
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _PatientHeroCard(
            name: controller.patientName.isEmpty ? 'Monitored Patient' : controller.patientName,
            age: controller.patientAge == null ? 'Age not set' : '${controller.patientAge} years',
            severity: severity,
            lastUpdated: _formatDateTime(controller.lastTransmissionAt),
          ),
          const SizedBox(height: 12),
          _StatusBanner(
            color: _severityColor(severity),
            title: _severityLabel(severity),
            message: statusText,
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(child: _StatCard(label: 'Risk Score', value: '$risk%')),
              const SizedBox(width: 10),
              Expanded(child: _StatCard(label: 'Movement Activity', value: '${movement.toStringAsFixed(0)}%')),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: _StatCard(
                  label: 'Last Movement',
                  value: _formatDateTime(controller.lastTransmissionAt),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _MiniTrendCard(value: movement / 100),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: () => controller.refreshCaregiverData(),
                  icon: const Icon(Icons.visibility_outlined),
                  label: const Text('View Details'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(backgroundColor: const Color(0xFFC2453F)),
                  onPressed: openAlerts,
                  icon: const Icon(Icons.warning_amber_rounded),
                  label: const Text('Emergency Actions'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class LiveMonitoringScreen extends StatelessWidget {
  const LiveMonitoringScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  Widget build(BuildContext context) {
    final live = controller.liveStatus;
    final severity = live?.severity ?? 'low';
    final riskValue = (live?.score ?? controller.lastDetection?.score ?? 0).clamp(0.0, 1.0).toDouble();
    final fallValue = (live?.fallProbability ?? controller.lastDetection?.fallProbability ?? 0)
        .clamp(0.0, 1.0)
        .toDouble();

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        _StatusBanner(
          color: _severityColor(severity),
          title: 'Live Status: ${_severityLabel(severity)}',
          message: live?.lastMessage ?? 'Monitoring active.',
        ),
        const SizedBox(height: 16),
        const _WavePulseCard(),
        const SizedBox(height: 16),
        _CircularRiskMeter(label: 'Risk Level', value: riskValue),
        const SizedBox(height: 16),
        _StatCard(label: 'Movement Intensity', value: '${(fallValue * 100).toStringAsFixed(0)}%'),
        const SizedBox(height: 10),
        _StatCard(label: 'Stability', value: _stabilityText(riskValue)),
        const SizedBox(height: 10),
        _StatCard(label: 'Alert Level', value: _severityLabel(severity)),
        const SizedBox(height: 16),
        _LocationSection(controller: controller),
        const SizedBox(height: 16),
        const Text('Recent Timeline', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18)),
        const SizedBox(height: 8),
        _TimelineItem(label: _formatDateTime(DateTime.now().subtract(const Duration(minutes: 1))), text: 'Monitoring active'),
        _TimelineItem(label: _formatDateTime(controller.lastTransmissionAt), text: 'Latest movement analyzed'),
        _TimelineItem(label: _formatDateTime(DateTime.now()), text: live?.lastMessage ?? 'No abnormal movement'),
      ],
    );
  }
}

class AlertsScreen extends StatelessWidget {
  const AlertsScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  Widget build(BuildContext context) {
    final alerts = controller.caregiverAlerts;
    return RefreshIndicator(
      onRefresh: () => controller.refreshCaregiverData(),
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          if (alerts.isEmpty)
            const _StatusBanner(
              color: Color(0xFF1B9B8B),
              title: 'No Active Alerts',
              message: 'Everything looks stable right now.',
            ),
          ...alerts.map(
            (alert) => Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: _AlertCard(
                alert: alert,
                onAcknowledge: () => controller.acknowledgeAlert(alert.id),
                onResolve: () => controller.resolveAlert(alert.id),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class InsightsScreen extends StatelessWidget {
  const InsightsScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  Widget build(BuildContext context) {
    final risk = ((controller.liveStatus?.score ?? 0) * 100).round();
    final movement = ((controller.liveStatus?.fallProbability ?? 0) * 100).round();
    final summary = controller.summary;
    final insight = movement < 25
        ? 'Patient has been less active than usual today.'
        : movement > 70
            ? 'Higher instability trends detected in recent hours.'
            : 'Movement pattern appears balanced today.';

    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        const Text('Daily & Weekly Insights', style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700)),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(child: _StatCard(label: 'Activity Level', value: '$movement%')),
            const SizedBox(width: 10),
            Expanded(child: _StatCard(label: 'Risk Pattern', value: '$risk%')),
          ],
        ),
        const SizedBox(height: 12),
        _SimpleBarGraph(values: [
          (movement / 100).clamp(0.1, 1.0),
          ((movement + 10) / 100).clamp(0.1, 1.0),
          ((movement - 5) / 100).clamp(0.1, 1.0),
          ((movement + 8) / 100).clamp(0.1, 1.0),
          (risk / 100).clamp(0.1, 1.0),
          ((risk - 7) / 100).clamp(0.1, 1.0),
          ((risk + 4) / 100).clamp(0.1, 1.0),
        ]),
        const SizedBox(height: 12),
        _StatusBanner(
          color: const Color(0xFF2A7DA8),
          title: 'AI Insight',
          message: insight,
        ),
        if (summary != null) ...[
          const SizedBox(height: 12),
          _StatusBanner(
            color: const Color(0xFF4A708F),
            title: 'Monitoring Summary',
            message:
                '${summary.activeSessions} active monitoring sessions and ${summary.openAlerts} open alerts across ${summary.totalPatients} patients.',
          ),
        ],
      ],
    );
  }
}

class PatientProfileScreen extends StatefulWidget {
  const PatientProfileScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  State<PatientProfileScreen> createState() => _PatientProfileScreenState();
}

class _PatientProfileScreenState extends State<PatientProfileScreen> {
  Future<void> _editProfile() async {
    final controller = widget.controller;
    final nameCtrl = TextEditingController(text: controller.patientName);
    final ageCtrl = TextEditingController(text: controller.patientAge?.toString() ?? '');
    final noteCtrl = TextEditingController(text: controller.medicalNotes);
    final contactCtrl = TextEditingController(text: controller.emergencyContact);

    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (context) {
        return Padding(
          padding: EdgeInsets.fromLTRB(16, 16, 16, 16 + MediaQuery.of(context).viewInsets.bottom),
          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text('Update Patient Profile', style: TextStyle(fontWeight: FontWeight.w700, fontSize: 18)),
                const SizedBox(height: 12),
                TextField(controller: nameCtrl, decoration: const InputDecoration(labelText: 'Name')),
                const SizedBox(height: 8),
                TextField(controller: ageCtrl, decoration: const InputDecoration(labelText: 'Age')),
                const SizedBox(height: 8),
                TextField(controller: noteCtrl, decoration: const InputDecoration(labelText: 'Medical notes')),
                const SizedBox(height: 8),
                TextField(
                  controller: contactCtrl,
                  decoration: const InputDecoration(labelText: 'Emergency contact'),
                ),
                const SizedBox(height: 12),
                FilledButton(
                  onPressed: () async {
                    await controller.updateProfile(
                      patientName: nameCtrl.text,
                      patientAgeText: ageCtrl.text,
                      medicalNotes: noteCtrl.text,
                      emergencyContact: contactCtrl.text,
                    );
                    if (context.mounted) Navigator.pop(context);
                  },
                  child: const Text('Save'),
                ),
              ],
            ),
          ),
        );
      },
    );

    nameCtrl.dispose();
    ageCtrl.dispose();
    noteCtrl.dispose();
    contactCtrl.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          decoration: _cardDecoration(),
          child: Row(
            children: [
              const CircleAvatar(radius: 28, child: Icon(Icons.person_outline)),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      controller.patientName.isEmpty ? 'Monitored Patient' : controller.patientName,
                      style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 18),
                    ),
                    Text(controller.patientAge == null ? 'Age not set' : '${controller.patientAge} years'),
                  ],
                ),
              ),
              IconButton(
                onPressed: _editProfile,
                icon: const Icon(Icons.edit_outlined),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        _StatCard(label: 'Medical Notes', value: controller.medicalNotes.isEmpty ? 'Not provided' : controller.medicalNotes),
        const SizedBox(height: 10),
        _StatCard(
          label: 'Emergency Contact',
          value: controller.emergencyContact.isEmpty ? 'Not provided' : controller.emergencyContact,
        ),
        const SizedBox(height: 10),
        _StatCard(
          label: 'Device Status',
          value: controller.isStreaming ? 'Monitoring active' : 'Monitoring paused',
        ),
        const SizedBox(height: 10),
        _StatCard(
          label: 'Home Location',
          value: controller.hasHomeLocation
              ? '${controller.homeLatitude!.toStringAsFixed(5)}, ${controller.homeLongitude!.toStringAsFixed(5)}'
              : 'Not set yet',
        ),
        const SizedBox(height: 10),
        Wrap(
          spacing: 10,
          runSpacing: 10,
          children: [
            FilledButton.icon(
              onPressed: controller.setHomeLocationFromCurrent,
              icon: const Icon(Icons.home_outlined),
              label: const Text('Set Home from Current'),
            ),
            OutlinedButton.icon(
              onPressed: controller.hasHomeLocation ? controller.clearHomeLocation : null,
              icon: const Icon(Icons.delete_outline),
              label: const Text('Clear Home'),
            ),
          ],
        ),
        if (controller.locationError != null) ...[
          const SizedBox(height: 10),
          _StatusBanner(
            color: const Color(0xFFB53B34),
            title: 'Location Notice',
            message: controller.locationError!,
          ),
        ],
      ],
    );
  }
}

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.controller});

  final MonitoringController controller;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _emailController;

  @override
  void initState() {
    super.initState();
    _emailController = TextEditingController(text: widget.controller.caregiverEmail);
  }

  @override
  void didUpdateWidget(covariant SettingsScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    final latest = widget.controller.caregiverEmail;
    if (_emailController.text != latest) {
      _emailController.text = latest;
    }
  }

  @override
  void dispose() {
    _emailController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = widget.controller;
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        SwitchListTile(
          title: const Text('Notifications'),
          subtitle: const Text('Receive immediate risk and emergency updates'),
          value: controller.notificationsEnabled,
          onChanged: controller.setNotificationsEnabled,
        ),
        const SizedBox(height: 8),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: _cardDecoration(),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Alert Delivery', style: TextStyle(fontWeight: FontWeight.w700)),
              const SizedBox(height: 8),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Email Alerts'),
                subtitle: const Text('Send critical alerts to caregiver email'),
                value: controller.alertViaEmail,
                onChanged: controller.notificationsEnabled ? controller.setAlertViaEmail : null,
              ),
              if (controller.alertViaEmail) ...[
                TextField(
                  controller: _emailController,
                  keyboardType: TextInputType.emailAddress,
                  decoration: const InputDecoration(
                    labelText: 'Caregiver Email',
                    hintText: 'caregiver@example.com',
                  ),
                  onSubmitted: controller.setCaregiverEmail,
                ),
                const SizedBox(height: 8),
                Align(
                  alignment: Alignment.centerRight,
                  child: FilledButton(
                    onPressed: () => controller.setCaregiverEmail(_emailController.text),
                    child: const Text('Save Email'),
                  ),
                ),
              ],
              const SizedBox(height: 6),
              SwitchListTile(
                contentPadding: EdgeInsets.zero,
                title: const Text('Alarm Sound'),
                subtitle: const Text('Play loud in-app alarm for severe alerts'),
                value: controller.alertViaAlarm,
                onChanged: controller.notificationsEnabled ? controller.setAlertViaAlarm : null,
              ),
              if (controller.alertViaAlarm && controller.notificationsEnabled)
                Align(
                  alignment: Alignment.centerRight,
                  child: TextButton.icon(
                    onPressed: controller.triggerTestAlarm,
                    icon: const Icon(Icons.play_arrow_rounded),
                    label: const Text('Test Alarm'),
                  ),
                ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: _cardDecoration(),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Alert Sensitivity', style: TextStyle(fontWeight: FontWeight.w700)),
              const SizedBox(height: 10),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'low', label: Text('Low')),
                  ButtonSegment(value: 'medium', label: Text('Medium')),
                  ButtonSegment(value: 'high', label: Text('High')),
                ],
                selected: {controller.alertSensitivity},
                onSelectionChanged: (value) => controller.setAlertSensitivity(value.first),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        SwitchListTile(
          title: const Text('Live Location Tracking'),
          subtitle: const Text('Share location for caregiver visibility and patient guidance'),
          value: controller.locationTrackingEnabled,
          onChanged: (enabled) {
            if (enabled) {
              controller.startLocationTracking();
            } else {
              controller.stopLocationTracking();
            }
          },
        ),
        if (controller.locationError != null)
          _StatusBanner(
            color: const Color(0xFFB53B34),
            title: 'Location Access',
            message: controller.locationError!,
          ),
        const SizedBox(height: 12),
        FilledButton.icon(
          onPressed: controller.isStreaming ? controller.stopMonitoring : controller.startMonitoring,
          icon: Icon(controller.isStreaming ? Icons.pause_circle_outline : Icons.play_circle_outline),
          label: Text(controller.isStreaming ? 'Pause Monitoring' : 'Activate Monitoring'),
        ),
        const SizedBox(height: 10),
        OutlinedButton.icon(
          onPressed: controller.caregiverLogout,
          icon: const Icon(Icons.logout),
          label: const Text('Sign Out Caregiver'),
        ),
      ],
    );
  }
}

class PatientModeHome extends StatelessWidget {
  const PatientModeHome({super.key, required this.controller});

  final MonitoringController controller;

  @override
  Widget build(BuildContext context) {
    final severity = controller.liveStatus?.severity ?? 'low';
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Container(
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            color: _severityColor(severity).withValues(alpha: 0.12),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('You are safe', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 26)),
              const SizedBox(height: 8),
              Text(controller.isStreaming ? 'Monitoring is active' : 'Monitoring will resume shortly'),
            ],
          ),
        ),
        if (controller.currentPosition != null) ...[
          const SizedBox(height: 14),
          _LocationMapCard(
            current: LatLng(controller.currentPosition!.latitude, controller.currentPosition!.longitude),
            home: controller.hasHomeLocation
                ? LatLng(controller.homeLatitude!, controller.homeLongitude!)
                : null,
            compact: false,
          ),
        ],
        if (controller.currentPosition != null && controller.hasHomeLocation) ...[
          const SizedBox(height: 10),
          _StatusBanner(
            color: const Color(0xFF2A7DA8),
            title: 'Direction Home',
            message:
                '${_distanceKm(controller.currentPosition!.latitude, controller.currentPosition!.longitude, controller.homeLatitude!, controller.homeLongitude!).toStringAsFixed(2)} km away, head ${_bearingDirection(controller.currentPosition!.latitude, controller.currentPosition!.longitude, controller.homeLatitude!, controller.homeLongitude!)}.',
          ),
        ],
        if (controller.currentPosition == null || !controller.hasHomeLocation) ...[
          const SizedBox(height: 12),
          const _StatusBanner(
            color: Color(0xFF2A7DA8),
            title: 'Location Guidance',
            message: 'Ask your caregiver to set Home Location from the profile for navigation help.',
          ),
        ],
        const SizedBox(height: 18),
        FilledButton.icon(
          style: FilledButton.styleFrom(
            backgroundColor: const Color(0xFFC2453F),
            padding: const EdgeInsets.symmetric(vertical: 18),
          ),
          onPressed: () => _confirmEmergency(context),
          icon: const Icon(Icons.sos_outlined),
          label: const Text('Call for Help'),
        ),
        const SizedBox(height: 12),
        OutlinedButton.icon(
          onPressed: controller.emergencyContact.isEmpty ? null : () {},
          icon: const Icon(Icons.phone_outlined),
          label: const Text('Call Caregiver'),
        ),
      ],
    );
  }

  Future<void> _confirmEmergency(BuildContext context) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Send emergency alert?'),
          content: const Text('This will immediately notify the caregiver.'),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
            FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Send Alert')),
          ],
        );
      },
    );
    if (confirmed == true) {
      await controller.triggerEmergencyAlert();
    }
  }
}

class _PatientHeroCard extends StatelessWidget {
  const _PatientHeroCard({
    required this.name,
    required this.age,
    required this.severity,
    required this.lastUpdated,
  });

  final String name;
  final String age;
  final String severity;
  final String lastUpdated;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        gradient: const LinearGradient(colors: [Color(0xFF0A7FA6), Color(0xFF155A9B)]),
        borderRadius: BorderRadius.circular(22),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(name, style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.w700)),
          const SizedBox(height: 6),
          Text(age, style: const TextStyle(color: Color(0xFFE3F4FB))),
          const SizedBox(height: 12),
          Row(
            children: [
              _StatusPill(label: _severityEmoji(severity), color: _severityColor(severity)),
              const SizedBox(width: 10),
              Text('Updated $lastUpdated', style: const TextStyle(color: Colors.white70)),
            ],
          ),
        ],
      ),
    );
  }
}

class _StatusBanner extends StatelessWidget {
  const _StatusBanner({required this.color, required this.title, required this.message});

  final Color color;
  final String title;
  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: TextStyle(color: color, fontWeight: FontWeight.w700)),
          const SizedBox(height: 6),
          Text(message),
        ],
      ),
    );
  }
}

class _StatusPill extends StatelessWidget {
  const _StatusPill({required this.label, required this.color});

  final String label;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
      decoration: BoxDecoration(color: color.withValues(alpha: 0.2), borderRadius: BorderRadius.circular(999)),
      child: Text(label, style: TextStyle(color: color, fontWeight: FontWeight.w700)),
    );
  }
}

class _StatCard extends StatelessWidget {
  const _StatCard({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(color: Color(0xFF5D7385))),
          const SizedBox(height: 8),
          Text(value, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
        ],
      ),
    );
  }
}

class _MiniTrendCard extends StatelessWidget {
  const _MiniTrendCard({required this.value});
  final double value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('24h Activity'),
          const SizedBox(height: 8),
          LinearProgressIndicator(value: value, minHeight: 10),
        ],
      ),
    );
  }
}

class _WavePulseCard extends StatelessWidget {
  const _WavePulseCard();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: _cardDecoration(),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Motion Activity', style: TextStyle(fontWeight: FontWeight.w700)),
          SizedBox(height: 12),
          Row(
            children: [
              Expanded(child: _PulseBar(height: 14)),
              SizedBox(width: 6),
              Expanded(child: _PulseBar(height: 20)),
              SizedBox(width: 6),
              Expanded(child: _PulseBar(height: 30)),
              SizedBox(width: 6),
              Expanded(child: _PulseBar(height: 18)),
              SizedBox(width: 6),
              Expanded(child: _PulseBar(height: 12)),
            ],
          ),
        ],
      ),
    );
  }
}

class _PulseBar extends StatelessWidget {
  const _PulseBar({required this.height});
  final double height;

  @override
  Widget build(BuildContext context) {
    return Container(
      height: height,
      decoration: BoxDecoration(
        color: const Color(0xFF1788B3).withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(8),
      ),
    );
  }
}

class _CircularRiskMeter extends StatelessWidget {
  const _CircularRiskMeter({required this.label, required this.value});

  final String label;
  final double value;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: _cardDecoration(),
      child: Row(
        children: [
          SizedBox(
            width: 76,
            height: 76,
            child: CircularProgressIndicator(
              value: value,
              strokeWidth: 9,
              backgroundColor: Colors.grey.shade200,
            ),
          ),
          const SizedBox(width: 16),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(label, style: const TextStyle(fontWeight: FontWeight.w700)),
              const SizedBox(height: 4),
              Text('${(value * 100).toStringAsFixed(0)}%', style: const TextStyle(fontSize: 22)),
            ],
          ),
        ],
      ),
    );
  }
}

class _TimelineItem extends StatelessWidget {
  const _TimelineItem({required this.label, required this.text});
  final String label;
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: _cardDecoration(),
      child: Row(
        children: [
          Text(label, style: const TextStyle(color: Color(0xFF5D7385))),
          const SizedBox(width: 10),
          Expanded(child: Text(text)),
        ],
      ),
    );
  }
}

class _AlertCard extends StatelessWidget {
  const _AlertCard({
    required this.alert,
    required this.onAcknowledge,
    required this.onResolve,
  });

  final AlertRecordModel alert;
  final VoidCallback onAcknowledge;
  final VoidCallback onResolve;

  @override
  Widget build(BuildContext context) {
    final color = _severityColor(alert.severity);
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: _cardDecoration(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _StatusPill(label: _severityLabel(alert.severity), color: color),
              const Spacer(),
              Text(_formatDateTime(alert.createdAt), style: const TextStyle(color: Color(0xFF6A7B8A))),
            ],
          ),
          const SizedBox(height: 10),
          Text(alert.message, style: const TextStyle(fontWeight: FontWeight.w700)),
          const SizedBox(height: 6),
          Text(alert.manuallyTriggered ? 'Manual alert' : 'Automatic detection alert'),
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              OutlinedButton(onPressed: onAcknowledge, child: const Text('Acknowledge')),
              OutlinedButton(onPressed: onResolve, child: const Text('Resolve')),
              OutlinedButton(onPressed: () {}, child: const Text('Call Patient')),
              FilledButton(onPressed: () {}, child: const Text('Notify Contact')),
            ],
          ),
        ],
      ),
    );
  }
}

class _LocationSection extends StatelessWidget {
  const _LocationSection({required this.controller});

  final MonitoringController controller;

  @override
  Widget build(BuildContext context) {
    final current = controller.currentPosition;
    if (!controller.locationTrackingEnabled) {
      return Container(
        padding: const EdgeInsets.all(14),
        decoration: _cardDecoration(),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Live Location', style: TextStyle(fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            const Text('Location tracking is off.'),
            const SizedBox(height: 8),
            FilledButton.icon(
              onPressed: controller.startLocationTracking,
              icon: const Icon(Icons.location_searching_outlined),
              label: const Text('Enable Tracking'),
            ),
          ],
        ),
      );
    }

    if (current == null) {
      return Container(
        padding: const EdgeInsets.all(14),
        decoration: _cardDecoration(),
        child: const Text('Fetching live location...'),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _LocationMapCard(
          current: LatLng(current.latitude, current.longitude),
          home: controller.hasHomeLocation ? LatLng(controller.homeLatitude!, controller.homeLongitude!) : null,
          compact: false,
        ),
        const SizedBox(height: 8),
        _StatCard(
          label: 'Current Position',
          value: '${current.latitude.toStringAsFixed(5)}, ${current.longitude.toStringAsFixed(5)}',
        ),
      ],
    );
  }
}

class _LocationMapCard extends StatelessWidget {
  const _LocationMapCard({
    required this.current,
    required this.home,
    required this.compact,
  });

  final LatLng current;
  final LatLng? home;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final points = <LatLng>[if (home != null) home!, current];
    return Container(
      height: compact ? 200 : 260,
      clipBehavior: Clip.antiAlias,
      decoration: _cardDecoration(),
      child: FlutterMap(
        options: MapOptions(
          initialCenter: current,
          initialZoom: 15,
        ),
        children: [
          TileLayer(
            urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            userAgentPackageName: 'com.example.newapp',
          ),
          if (points.length == 2)
            PolylineLayer(
              polylines: [
                Polyline(
                  points: points,
                  strokeWidth: 4,
                  color: const Color(0xFF2A7DA8),
                ),
              ],
            ),
          MarkerLayer(
            markers: [
              Marker(
                point: current,
                width: 40,
                height: 40,
                child: const Icon(Icons.person_pin_circle, color: Color(0xFF155A9B), size: 34),
              ),
              if (home != null)
                Marker(
                  point: home!,
                  width: 40,
                  height: 40,
                  child: const Icon(Icons.home_rounded, color: Color(0xFF1B9B8B), size: 30),
                ),
            ],
          ),
        ],
      ),
    );
  }
}

class _SimpleBarGraph extends StatelessWidget {
  const _SimpleBarGraph({required this.values});
  final List<double> values;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: _cardDecoration(),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: values
            .map(
              (value) => Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4),
                  child: Container(
                    height: 90 * value.clamp(0.1, 1.0),
                    decoration: BoxDecoration(
                      color: const Color(0xFF2B88B3),
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                ),
              ),
            )
            .toList(),
      ),
    );
  }
}

BoxDecoration _cardDecoration() {
  return BoxDecoration(
    color: Colors.white,
    borderRadius: BorderRadius.circular(16),
    boxShadow: const [BoxShadow(color: Color(0x140F2E4D), blurRadius: 14, offset: Offset(0, 6))],
  );
}

Color _severityColor(String severity) {
  switch (severity) {
    case 'medium':
      return const Color(0xFFF0A542);
    case 'high_risk':
      return const Color(0xFFDE6B48);
    case 'fall_detected':
      return const Color(0xFFB53B34);
    default:
      return const Color(0xFF1B9B8B);
  }
}

String _severityLabel(String severity) {
  switch (severity) {
    case 'high_risk':
      return 'High Risk';
    case 'fall_detected':
      return 'Fall Detected';
    case 'medium':
      return 'Medium Risk';
    default:
      return 'Safe';
  }
}

String _severityEmoji(String severity) {
  switch (severity) {
    case 'high_risk':
      return '🔴 High Risk';
    case 'fall_detected':
      return '⚫ Fall Detected';
    case 'medium':
      return '🟡 Medium Risk';
    default:
      return '🟢 Safe';
  }
}

String _formatDateTime(DateTime? value) {
  if (value == null) return 'just now';
  final local = value.toLocal();
  final h = local.hour.toString().padLeft(2, '0');
  final m = local.minute.toString().padLeft(2, '0');
  return '$h:$m';
}

String _stabilityText(double riskValue) {
  if (riskValue > 0.75) return 'Unstable';
  if (riskValue > 0.45) return 'Observe closely';
  return 'Stable';
}

double _distanceKm(double lat1, double lon1, double lat2, double lon2) {
  const distance = Distance();
  return distance.as(LengthUnit.Kilometer, LatLng(lat1, lon1), LatLng(lat2, lon2));
}

String _bearingDirection(double lat1, double lon1, double lat2, double lon2) {
  const distance = Distance();
  final bearing = distance.bearing(LatLng(lat1, lon1), LatLng(lat2, lon2));
  final normalized = (bearing + 360) % 360;
  if (normalized >= 337.5 || normalized < 22.5) return 'North';
  if (normalized < 67.5) return 'North-East';
  if (normalized < 112.5) return 'East';
  if (normalized < 157.5) return 'South-East';
  if (normalized < 202.5) return 'South';
  if (normalized < 247.5) return 'South-West';
  if (normalized < 292.5) return 'West';
  return 'North-West';
}
