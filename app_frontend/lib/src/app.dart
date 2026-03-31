import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'models.dart';
import 'monitoring_controller.dart';

class ElderlyMonitorApp extends StatefulWidget {
  const ElderlyMonitorApp({
    super.key,
    this.controller,
  });

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
      title: 'Elderly Monitor',
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF4EEE4),
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF0F8579),
          brightness: Brightness.light,
        ),
        textTheme: ThemeData.light().textTheme.apply(
              bodyColor: const Color(0xFF163126),
              displayColor: const Color(0xFF163126),
            ),
        cardTheme: CardThemeData(
          color: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(24),
            side: const BorderSide(color: Color(0xFFE4D8C8)),
          ),
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: const Color(0xFFF8F4EE),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(18),
            borderSide: const BorderSide(color: Color(0xFFD9CBB9)),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(18),
            borderSide: const BorderSide(color: Color(0xFFD9CBB9)),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(18),
            borderSide: const BorderSide(
              color: Color(0xFF0F8579),
              width: 1.4,
            ),
          ),
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 16,
            vertical: 16,
          ),
        ),
      ),
      home: FutureBuilder<void>(
        future: _initializationFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState != ConnectionState.done) {
            return const _LoadingScreen();
          }
          return MonitoringHomePage(controller: _controller);
        },
      ),
    );
  }
}

class MonitoringHomePage extends StatefulWidget {
  const MonitoringHomePage({
    super.key,
    required this.controller,
  });

  final MonitoringController controller;

  @override
  State<MonitoringHomePage> createState() => _MonitoringHomePageState();
}

class _MonitoringHomePageState extends State<MonitoringHomePage> {
  late final TextEditingController _backendUrlController;
  late final TextEditingController _patientNameController;
  late final TextEditingController _patientAgeController;
  late final TextEditingController _roomLabelController;
  late final TextEditingController _deviceLabelController;

  @override
  void initState() {
    super.initState();
    _backendUrlController = TextEditingController(text: widget.controller.backendUrl);
    _patientNameController = TextEditingController(text: widget.controller.patientName);
    _patientAgeController = TextEditingController(
      text: widget.controller.patientAge?.toString() ?? '',
    );
    _roomLabelController = TextEditingController(text: widget.controller.roomLabel);
    _deviceLabelController = TextEditingController(text: widget.controller.deviceLabel);
  }

  @override
  void dispose() {
    _backendUrlController.dispose();
    _patientNameController.dispose();
    _patientAgeController.dispose();
    _roomLabelController.dispose();
    _deviceLabelController.dispose();
    super.dispose();
  }

  Future<void> _saveSetup() async {
    await widget.controller.saveSetup(
      backendUrl: _backendUrlController.text,
      patientName: _patientNameController.text,
      patientAgeText: _patientAgeController.text,
      roomLabel: _roomLabelController.text,
      deviceLabel: _deviceLabelController.text,
    );
  }

  Future<void> _saveAndStart() async {
    await _saveSetup();
    if (widget.controller.lastError == null) {
      await widget.controller.startMonitoring();
    }
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.controller,
      builder: (context, _) {
        final controller = widget.controller;
        return Scaffold(
          body: Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: <Color>[
                  Color(0xFFE2F2EA),
                  Color(0xFFF4EEE4),
                  Color(0xFFF9F6F1),
                ],
              ),
            ),
            child: SafeArea(
              child: Stack(
                children: <Widget>[
                  SingleChildScrollView(
                    padding: const EdgeInsets.fromLTRB(20, 18, 20, 28),
                    child: Center(
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(maxWidth: 760),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: <Widget>[
                            _HeroPanel(controller: controller),
                            const SizedBox(height: 18),
                            if (controller.lastError != null) ...<Widget>[
                              _BannerMessage(
                                title: 'Attention Needed',
                                message: controller.lastError!,
                                backgroundColor: const Color(0xFFFCE2DF),
                                accentColor: const Color(0xFFB53B34),
                              ),
                              const SizedBox(height: 14),
                            ],
                            _BannerMessage(
                              title: 'Live Status',
                              message: controller.statusMessage,
                              backgroundColor: const Color(0xFFE8F4F1),
                              accentColor: const Color(0xFF0F8579),
                            ),
                            const SizedBox(height: 18),
                            _buildSetupCard(controller),
                            const SizedBox(height: 18),
                            _buildSensorAccessCard(controller),
                            const SizedBox(height: 18),
                            _buildSessionCard(controller),
                            const SizedBox(height: 18),
                            _buildDetectionCard(controller),
                            const SizedBox(height: 18),
                            _buildEmergencyCard(controller),
                          ],
                        ),
                      ),
                    ),
                  ),
                  if (controller.isBusy)
                    const Positioned(
                      left: 0,
                      right: 0,
                      top: 0,
                      child: LinearProgressIndicator(minHeight: 3),
                    ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildSetupCard(MonitoringController controller) {
    return _CardShell(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          const _SectionHeader(
            title: 'Patient Setup',
            subtitle:
                'Save the phone configuration that will be used for live motion detection.',
          ),
          const SizedBox(height: 18),
          LayoutBuilder(
            builder: (context, constraints) {
              final useTwoColumns = constraints.maxWidth > 640;
              if (useTwoColumns) {
                return Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    Expanded(child: _buildPrimarySetupFields(controller)),
                    const SizedBox(width: 14),
                    Expanded(child: _buildSecondarySetupFields(controller)),
                  ],
                );
              }
              return Column(
                children: <Widget>[
                  _buildPrimarySetupFields(controller),
                  const SizedBox(height: 14),
                  _buildSecondarySetupFields(controller),
                ],
              );
            },
          ),
        ],
      ),
    );
  }

  Widget _buildSessionCard(MonitoringController controller) {
    return _CardShell(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          const _SectionHeader(
            title: 'Live Session',
            subtitle:
                'Start or stop sensor streaming, and keep an eye on the current backend session.',
          ),
          const SizedBox(height: 18),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: <Widget>[
              _MetricChip(
                label: 'Patient ID',
                value: controller.patientId ?? 'Not created yet',
              ),
              _MetricChip(
                label: 'Device ID',
                value: controller.deviceId ?? 'Not created yet',
              ),
              _MetricChip(
                label: 'Session ID',
                value: controller.sessionId ?? 'No active session',
              ),
              _MetricChip(
                label: 'Last Upload',
                value: _formatTimestamp(controller.lastTransmissionAt),
              ),
            ],
          ),
          const SizedBox(height: 18),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: <Widget>[
              FilledButton.icon(
                onPressed:
                    controller.isStreaming || controller.isBusy ? null : _saveAndStart,
                icon: const Icon(Icons.play_arrow_rounded),
                label: const Text('Start Monitoring'),
              ),
              OutlinedButton.icon(
                onPressed: controller.isStreaming && !controller.isBusy
                    ? controller.stopMonitoring
                    : null,
                icon: const Icon(Icons.stop_circle_outlined),
                label: const Text('Stop Monitoring'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildSensorAccessCard(MonitoringController controller) {
    final sensorStatus = controller.sensorAccessStatus;
    final latestTelemetry = controller.latestTelemetry;

    return _CardShell(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          const _SectionHeader(
            title: 'Sensor Access',
            subtitle:
                'Check that the phone can read the accelerometer and gyroscope before starting live monitoring.',
          ),
          const SizedBox(height: 18),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: <Widget>[
              _SensorStatusTile(
                label: 'Accelerometer',
                isReady: sensorStatus?.accelerometerAvailable ?? false,
              ),
              _SensorStatusTile(
                label: 'Gyroscope',
                isReady: sensorStatus?.gyroscopeAvailable ?? false,
              ),
              _MetricChip(
                label: 'Last Checked',
                value: sensorStatus == null
                    ? 'Not checked'
                    : _formatTimestamp(sensorStatus.checkedAt),
              ),
              _MetricChip(
                label: 'Latest Batch',
                value: latestTelemetry == null
                    ? 'No telemetry yet'
                    : '${latestTelemetry.samplesInLastBatch} samples',
              ),
            ],
          ),
          const SizedBox(height: 16),
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: <Widget>[
              FilledButton.icon(
                onPressed: controller.isBusy
                    ? null
                    : () => controller.refreshSensorStatus(),
                icon: const Icon(Icons.sensors_outlined),
                label: const Text('Check Sensors'),
              ),
              if (latestTelemetry != null)
                OutlinedButton.icon(
                  onPressed: null,
                  icon: const Icon(Icons.memory_outlined),
                  label: Text(
                    'Source ${latestTelemetry.source}',
                  ),
                ),
            ],
          ),
          const SizedBox(height: 14),
          const Text(
            'On some devices the operating system shows the motion access prompt the first time sensor streams are touched. Keep the app open and allow access if prompted.',
            style: TextStyle(
              color: Color(0xFF51645C),
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDetectionCard(MonitoringController controller) {
    return _CardShell(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          const _SectionHeader(
            title: 'Risk Detection',
            subtitle:
                'Live severity updates from the FastAPI backend appear here after sensor batches are processed.',
          ),
          const SizedBox(height: 18),
          _DetectionPanel(
            detection: controller.lastDetection,
            liveStatus: controller.liveStatus,
          ),
        ],
      ),
    );
  }

  Widget _buildEmergencyCard(MonitoringController controller) {
    return _CardShell(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: <Widget>[
          const _SectionHeader(
            title: 'Emergency Trigger',
            subtitle:
                'Use this when the user needs immediate attention regardless of automatic classification.',
          ),
          const SizedBox(height: 18),
          _EmergencyPanel(controller: controller),
        ],
      ),
    );
  }

  Widget _buildPrimarySetupFields(MonitoringController controller) {
    return Column(
      children: <Widget>[
        TextField(
          controller: _backendUrlController,
          enabled: !controller.isStreaming,
          keyboardType: TextInputType.url,
          decoration: const InputDecoration(
            labelText: 'Backend URL',
            hintText: 'http://10.0.2.2:8000',
          ),
        ),
        const SizedBox(height: 12),
        TextField(
          controller: _patientNameController,
          enabled: !controller.isStreaming,
          decoration: const InputDecoration(
            labelText: 'Patient Name',
            hintText: 'Abdul Hameed',
          ),
        ),
        const SizedBox(height: 12),
        TextField(
          controller: _patientAgeController,
          enabled: !controller.isStreaming,
          keyboardType: TextInputType.number,
          inputFormatters: <TextInputFormatter>[
            FilteringTextInputFormatter.digitsOnly,
          ],
          decoration: const InputDecoration(
            labelText: 'Patient Age',
            hintText: '72',
          ),
        ),
      ],
    );
  }

  Widget _buildSecondarySetupFields(MonitoringController controller) {
    return Column(
      children: <Widget>[
        TextField(
          controller: _roomLabelController,
          enabled: !controller.isStreaming,
          decoration: const InputDecoration(
            labelText: 'Room Label',
            hintText: 'Room 204',
          ),
        ),
        const SizedBox(height: 12),
        TextField(
          controller: _deviceLabelController,
          enabled: !controller.isStreaming,
          decoration: const InputDecoration(
            labelText: 'Device Label',
            hintText: 'Ward Phone A',
          ),
        ),
        const SizedBox(height: 12),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: const Color(0xFFF8F4EE),
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: const Color(0xFFD9CBB9)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              const Text(
                'Connection Summary',
                style: TextStyle(
                  fontWeight: FontWeight.w700,
                  color: Color(0xFF163126),
                ),
              ),
              const SizedBox(height: 10),
              _InfoRow(
                label: 'Backend',
                value: controller.backendReachable ? 'Reachable' : 'Offline or not checked',
              ),
              _InfoRow(
                label: 'Ready',
                value: controller.isReady ? 'Yes' : 'Not yet',
              ),
              _InfoRow(
                label: 'Streaming',
                value: controller.isStreaming ? 'Active' : 'Stopped',
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        Wrap(
          spacing: 10,
          runSpacing: 10,
          children: <Widget>[
            FilledButton.icon(
              onPressed: controller.isStreaming || controller.isBusy ? null : _saveSetup,
              icon: const Icon(Icons.save_outlined),
              label: const Text('Save Setup'),
            ),
            OutlinedButton.icon(
              onPressed: controller.isBusy
                  ? null
                  : () => controller.refreshBackendReachability(),
              icon: const Icon(Icons.health_and_safety_outlined),
              label: const Text('Check Backend'),
            ),
          ],
        ),
        const SizedBox(height: 14),
        const Text(
          'Tip: use 10.0.2.2 for the Android emulator, 127.0.0.1 for the iOS simulator, and your computer\'s LAN IP for a physical phone.',
          style: TextStyle(
            color: Color(0xFF51645C),
            height: 1.4,
          ),
        ),
      ],
    );
  }
}

class _LoadingScreen extends StatelessWidget {
  const _LoadingScreen();

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: CircularProgressIndicator(),
      ),
    );
  }
}

class _HeroPanel extends StatelessWidget {
  const _HeroPanel({required this.controller});

  final MonitoringController controller;

  @override
  Widget build(BuildContext context) {
    final statusColor = _severityColor(
      controller.liveStatus?.severity ?? controller.lastDetection?.severity ?? 'low',
    );

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(30),
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: <Color>[
            Color(0xFF0F8579),
            Color(0xFF174B6D),
          ],
        ),
        boxShadow: const <BoxShadow>[
          BoxShadow(
            color: Color(0x1F174B6D),
            blurRadius: 24,
            offset: Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    const Text(
                      'Elderly Monitor',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 28,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      controller.patientName.isEmpty
                          ? 'Set up the patient profile to begin live mobile monitoring.'
                          : 'Live detection profile for ${controller.patientName}${controller.roomLabel.isEmpty ? '' : ' - ${controller.roomLabel}'}',
                      style: const TextStyle(
                        color: Color(0xFFF0F7F4),
                        height: 1.5,
                        fontSize: 15,
                      ),
                    ),
                  ],
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.16),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: <Widget>[
                    Container(
                      width: 10,
                      height: 10,
                      decoration: BoxDecoration(
                        color: statusColor,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      _severityLabel(
                        controller.liveStatus?.severity ??
                            controller.lastDetection?.severity ??
                            'low',
                      ),
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 22),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: <Widget>[
              _HeroStat(
                label: 'Backend',
                value: controller.backendReachable ? 'Online' : 'Offline',
              ),
              _HeroStat(
                label: 'Session',
                value: controller.isStreaming ? 'Streaming' : 'Idle',
              ),
              _HeroStat(
                label: 'Batches',
                value: controller.batchesSent.toString(),
              ),
              _HeroStat(
                label: 'Last batch',
                value: controller.lastBatchSize == 0
                    ? 'None'
                    : '${controller.lastBatchSize} samples',
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _DetectionPanel extends StatelessWidget {
  const _DetectionPanel({
    required this.detection,
    required this.liveStatus,
  });

  final DetectionResultModel? detection;
  final LiveStatusModel? liveStatus;

  @override
  Widget build(BuildContext context) {
    final effectiveSeverity = liveStatus?.severity ?? detection?.severity ?? 'low';
    final severityColor = _severityColor(effectiveSeverity);
    final message = liveStatus?.lastMessage ??
        detection?.message ??
        'No sensor batches have been analyzed yet.';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            color: severityColor.withValues(alpha: 0.12),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              Row(
                children: <Widget>[
                  Container(
                    width: 12,
                    height: 12,
                    decoration: BoxDecoration(
                      color: severityColor,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Text(
                    _severityLabel(effectiveSeverity),
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                      color: severityColor,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(
                message,
                style: const TextStyle(
                  fontSize: 15,
                  height: 1.45,
                  color: Color(0xFF24463A),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 18),
        Row(
          children: <Widget>[
            Expanded(
              child: _ScorePanel(
                title: 'Risk Score',
                value: detection?.score ?? liveStatus?.score ?? 0,
                color: severityColor,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _ScorePanel(
                title: 'Fall Probability',
                value: detection?.fallProbability ?? liveStatus?.fallProbability ?? 0,
                color: const Color(0xFFCC5A3C),
              ),
            ),
          ],
        ),
        const SizedBox(height: 18),
        Wrap(
          spacing: 12,
          runSpacing: 12,
          children: <Widget>[
            _MetricChip(
              label: 'Peak Acceleration',
              value: detection == null ? 'Pending' : '${detection!.peakAccG.toStringAsFixed(2)} g',
            ),
            _MetricChip(
              label: 'Peak Gyroscope',
              value: detection == null
                  ? 'Pending'
                  : '${detection!.peakGyroDps.toStringAsFixed(1)} dps',
            ),
            _MetricChip(
              label: 'Peak Jerk',
              value: detection == null
                  ? 'Pending'
                  : '${detection!.peakJerkGps.toStringAsFixed(2)} g/s',
            ),
            _MetricChip(
              label: 'Stillness Ratio',
              value: detection == null
                  ? 'Pending'
                  : '${(detection!.stillnessRatio * 100).toStringAsFixed(0)}%',
            ),
          ],
        ),
        const SizedBox(height: 18),
        Text(
          'Detector Reasons',
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.w700,
              ),
        ),
        const SizedBox(height: 10),
        if (detection == null || detection!.reasons.isEmpty)
          const Text(
            'Reasons will appear after the backend has analyzed a batch.',
            style: TextStyle(color: Color(0xFF64776F)),
          )
        else
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: detection!.reasons
                .map(
                  (reason) => Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                    decoration: BoxDecoration(
                      color: const Color(0xFFF2EFE9),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Text(reason),
                  ),
                )
                .toList(),
          ),
      ],
    );
  }
}

class _EmergencyPanel extends StatelessWidget {
  const _EmergencyPanel({required this.controller});

  final MonitoringController controller;

  @override
  Widget build(BuildContext context) {
    final alert = controller.activeAlert;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        Container(
          padding: const EdgeInsets.all(18),
          decoration: BoxDecoration(
            color: const Color(0xFFFFF1EA),
            borderRadius: BorderRadius.circular(20),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              const Icon(
                Icons.warning_amber_rounded,
                color: Color(0xFFCC5A3C),
                size: 30,
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Use the manual trigger when a caregiver needs to raise an alert immediately, even if the automatic detector has not classified a fall yet.',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: const Color(0xFF5B3A2C),
                        height: 1.45,
                      ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        FilledButton.icon(
          style: FilledButton.styleFrom(
            backgroundColor: const Color(0xFFCC5A3C),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 16),
          ),
          onPressed: controller.isBusy ? null : controller.triggerEmergencyAlert,
          icon: const Icon(Icons.sos_outlined),
          label: const Text('Send Emergency Alert'),
        ),
        const SizedBox(height: 16),
        if (alert == null)
          const Text(
            'No active manual alert on this phone yet.',
            style: TextStyle(color: Color(0xFF64776F)),
          )
        else
          Column(
            children: <Widget>[
              _InfoRow(label: 'Alert ID', value: alert.id),
              _InfoRow(label: 'Severity', value: _severityLabel(alert.severity)),
              _InfoRow(label: 'Status', value: alert.status),
              _InfoRow(label: 'Message', value: alert.message),
            ],
          ),
      ],
    );
  }
}

class _ScorePanel extends StatelessWidget {
  const _ScorePanel({
    required this.title,
    required this.value,
    required this.color,
  });

  final String title;
  final double value;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final safeValue = value.clamp(0.0, 1.0).toDouble();

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFF8F4EE),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            title,
            style: const TextStyle(
              fontWeight: FontWeight.w700,
              color: Color(0xFF163126),
            ),
          ),
          const SizedBox(height: 10),
          Text(
            '${(safeValue * 100).toStringAsFixed(0)}%',
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w800,
            ),
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(999),
            child: LinearProgressIndicator(
              value: safeValue,
              minHeight: 10,
              backgroundColor: color.withValues(alpha: 0.14),
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
        ],
      ),
    );
  }
}

class _BannerMessage extends StatelessWidget {
  const _BannerMessage({
    required this.title,
    required this.message,
    required this.backgroundColor,
    required this.accentColor,
  });

  final String title;
  final String message;
  final Color backgroundColor;
  final Color accentColor;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: backgroundColor,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Icon(
            Icons.info_outline_rounded,
            color: accentColor,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  title,
                  style: TextStyle(
                    color: accentColor,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  message,
                  style: const TextStyle(
                    color: Color(0xFF364942),
                    height: 1.45,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _CardShell extends StatelessWidget {
  const _CardShell({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: child,
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({
    required this.title,
    required this.subtitle,
  });

  final String title;
  final String subtitle;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text(
          title,
          style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.w800,
              ),
        ),
        const SizedBox(height: 6),
        Text(
          subtitle,
          style: const TextStyle(
            color: Color(0xFF5F726A),
            height: 1.5,
          ),
        ),
      ],
    );
  }
}

class _MetricChip extends StatelessWidget {
  const _MetricChip({
    required this.label,
    required this.value,
  });

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      constraints: const BoxConstraints(minWidth: 140),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFFF8F4EE),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            label,
            style: const TextStyle(
              color: Color(0xFF6B7B74),
              fontSize: 12,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            value,
            style: const TextStyle(
              color: Color(0xFF163126),
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }
}

class _SensorStatusTile extends StatelessWidget {
  const _SensorStatusTile({
    required this.label,
    required this.isReady,
  });

  final String label;
  final bool isReady;

  @override
  Widget build(BuildContext context) {
    final color = isReady ? const Color(0xFF1B9B8B) : const Color(0xFFB53B34);

    return Container(
      constraints: const BoxConstraints(minWidth: 160),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFFF8F4EE),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          Container(
            width: 12,
            height: 12,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 10),
          Flexible(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(
                  label,
                  style: const TextStyle(
                    color: Color(0xFF163126),
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  isReady ? 'Connected' : 'Unavailable',
                  style: TextStyle(color: color),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  const _InfoRow({
    required this.label,
    required this.value,
  });

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          SizedBox(
            width: 120,
            child: Text(
              label,
              style: const TextStyle(
                color: Color(0xFF6B7B74),
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(
                color: Color(0xFF163126),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _HeroStat extends StatelessWidget {
  const _HeroStat({
    required this.label,
    required this.value,
  });

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 150,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.14),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            label,
            style: const TextStyle(
              color: Color(0xFFD5ECE6),
              fontSize: 12,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
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
    case 'low':
      return 'Low';
    case 'medium':
      return 'Medium';
    case 'high_risk':
      return 'High Risk';
    case 'fall_detected':
      return 'Fall Detected';
    default:
      return severity.isEmpty ? 'Low' : severity.replaceAll('_', ' ').toUpperCase();
  }
}

String _formatTimestamp(DateTime? value) {
  if (value == null) {
    return 'No uploads yet';
  }

  final local = value.toLocal();
  final hour = local.hour.toString().padLeft(2, '0');
  final minute = local.minute.toString().padLeft(2, '0');
  final second = local.second.toString().padLeft(2, '0');
  return '$hour:$minute:$second';
}
