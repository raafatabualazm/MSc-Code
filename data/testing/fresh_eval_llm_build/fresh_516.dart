@pragma('vm:entry-point')
List<String> enumerateTelemetryBranches(List<int> samples) {
  if (samples.isEmpty) return ['idle'];
  List<String> out = [];
  void walk(int index, String path) {
    if (index >= samples.length) {
      out.add(path);
      return;
    }
    for (int size = 1; size <= 2; size++) {
      if (index + size > samples.length) continue;
      String token = '';
      if (size == 2) {
        int a = samples[index], b = samples[index + 1];
        if ((a - b).abs() > 2) continue;
        int total = a + b;
        if (a == 0 || b == 0) token = 'flat$total';
        else if ((a > 0) == (b > 0)) token = 'sync$total';
        else token = 'flip$total';
      } else {
        int v = samples[index];
        if (v < -3) token = 'deep$v';
        else if (v < 0) token = 'neg$v';
        else if (v == 0) token = 'zero';
        else if (v > 3) token = 'peak$v';
        else token = 'pos$v';
      }
      walk(index + size, path.isEmpty ? token : '$path|$token');
    }
  }
  walk(0, '');
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(enumerateTelemetryBranches([]).toString() == '[idle]');
  assert(enumerateTelemetryBranches([1, 2]).toString() == '[pos1|pos2, sync3]');
  assert(enumerateTelemetryBranches([5, 1]).length == 1);
  print('All tests passed!');
}