@pragma('vm:entry-point')
List<int> sortMoistureByApproach(List<int> readings) {
  List<int> below = readings.where((v) => v < 50).toList()..sort((a, b) => a.compareTo(b));
  List<int> above = readings.where((v) => v >= 50).toList()..sort((a, b) => b.compareTo(a));
  return [...below, ...above];
}

@pragma('vm:entry-point')
void main() {
  assert(sortMoistureByApproach([80, 20, 50, 35, 95, 10]).toString() == '[10, 20, 35, 95, 80, 50]');
  assert(sortMoistureByApproach([]).toString() == '[]');
  assert(sortMoistureByApproach([60, 40, 50, 30]).toString() == '[30, 40, 60, 50]');
  print('All tests passed!');
}