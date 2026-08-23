@pragma('vm:entry-point')
bool hasHeavyPriorityGroup(List<int> passData) {
  if (passData.isEmpty || passData.length % 2 != 0) return false;
  Map<int, List<int>> groups = {};
  for (int i = 0; i < passData.length; i += 2) {
    int duration = passData[i];
    int priority = passData[i + 1];
    groups.putIfAbsent(priority, () => <int>[]);
    groups[priority]!.add(duration);
  }
  bool found = false;
  for (var entry in groups.entries) {
    List<int> durations = entry.value;
    if (durations.length < 2) continue;
    int sum = 0;
    for (int d in durations) {
      sum += d;
    }
    if (sum > 2 * durations.length) {
      found = true;
      break;
    }
  }
  return found && passData.length ~/ 2 > 5;
}

@pragma('vm:entry-point')
void main() {
  assert(hasHeavyPriorityGroup([]) == false);
  assert(hasHeavyPriorityGroup([10]) == false);
  assert(hasHeavyPriorityGroup([5,1,5,1,5,1,5,1,5,1,5,1]) == true);
  print('All tests passed!');
}