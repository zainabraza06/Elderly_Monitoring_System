import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:newapp/src/app.dart';

void main() {
  testWidgets('renders the monitoring app shell', (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues(<String, Object>{});

    await tester.pumpWidget(const ElderlyMonitorApp());
    await tester.pumpAndSettle();

    expect(find.text('Elderly Monitor'), findsOneWidget);
    expect(find.text('Patient Setup'), findsOneWidget);
    expect(find.text('Start Monitoring'), findsOneWidget);
  });
}
