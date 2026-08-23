@pragma('vm:entry-point')
List<String> binWifiSignalReadings(String readings, String delimiter) {
  if (readings.isEmpty) return [];
  final parts = readings.split(delimiter);
  final List<String> result = [];
  for (final part in parts) {
    final int dbm = int.parse(part.trim());
    String label;
    if (dbm >= -50) {
      label = 'Excellent';
    } else if (dbm >= -70) {
      label = 'Good';
    } else if (dbm >= -85) {
      label = 'Fair';
    } else {
      label = 'Poor';
    }
    result.add(label);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(binWifiSignalReadings('', ',').toString() == '[]');
  assert(binWifiSignalReadings('-45,-51,-70,-71,-85,-86', ',').toString() == '[Excellent, Good, Good, Fair, Fair, Poor]');
  assert(binWifiSignalReadings('-50|-100', '|').toString() == '[Excellent, Poor]');
  print('All tests passed!');
}