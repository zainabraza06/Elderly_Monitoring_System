import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:safestep_ai/src/app.dart';

void main() {
  testWidgets('renders the monitoring app shell', (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues(<String, Object>{});

    await tester.pumpWidget(const ElderlyMonitorApp());
    await tester.pumpAndSettle();

    expect(find.text('Who is using this phone?'), findsOneWidget);
    expect(find.text('I am the caregiver'), findsOneWidget);
  });
}
