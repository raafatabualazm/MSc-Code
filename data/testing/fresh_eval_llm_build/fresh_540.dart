@pragma('vm:entry-point')
List<String> deduplicateTrafficCycles(List<int> durations) {
  if (durations.isEmpty) return [];
  if (durations.length % 3 != 0) {
    return ["invalid sequence length: must be multiple of 3"];
  }
  List<String> result = [];
  int i = 0;
  while (i < durations.length) {
    int g = durations[i];
    int y = durations[i + 1];
    int r = durations[i + 2];
    if (g <= 0 || r <= 0) {
      return ["invalid green or red duration: must be positive"];
    }
    if (y != 3) {
      return ["invalid yellow duration: must be exactly 3"];
    }
    int count = 1;
    int j = i + 3;
    while (j + 2 < durations.length &&
           durations[j] == g &&
           durations[j + 1] == y &&
           durations[j + 2] == r) {
      count++;
      j += 3;
    }
    int cycleStart = (i ~/ 3) + 1;
    if (count == 1) {
      result.add("Cycle $cycleStart: G=$g,Y=$y,R=$r");
    } else {
      result.add("Cycles ${cycleStart}-${cycleStart + count - 1}: G=$g,Y=$y,R=$r (x$count)");
    }
    i = j;
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(deduplicateTrafficCycles([]).isEmpty);
  assert(deduplicateTrafficCycles([30, 3, 20]).length == 1);
  assert(deduplicateTrafficCycles([30, 3, 20, 30, 3, 20]).first.contains('x2'));
  print('All tests passed!');
}