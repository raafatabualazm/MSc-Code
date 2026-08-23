@pragma('vm:entry-point')
String? batteryLevelAfterDays(int totalDays) {
  if (totalDays < 0) return null;
  const int dailyDrain = 20;
  const int rechargeInterval = 7;
  int battery = 100;
  for (int d = 1; d <= totalDays; d++) {
    if (d % rechargeInterval == 0) {
      battery = 100;
    }
    battery -= dailyDrain;
    if (battery < 0) battery = 0;
  }
  return '$battery%';
}

@pragma('vm:entry-point')
void main() {
  assert(batteryLevelAfterDays(0) == '100%');
  assert(batteryLevelAfterDays(5) == '0%');
  assert(batteryLevelAfterDays(7) == '80%');
  print('All tests passed!');
}