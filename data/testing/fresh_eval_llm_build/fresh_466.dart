@pragma('vm:entry-point')
List<int> classifyNickelRoundingBias(List<int> amounts) {
  Map<int, int> driftCounts = {};
  int roundDown = 0;
  int exact = 0;
  int roundUp = 0;
  for (final value in amounts) {
    int remainder = value.abs() % 5;
    int drift;
    if (remainder == 0) {
      drift = 0;
      exact++;
    } else if (remainder <= 2) {
      drift = -remainder;
      roundDown++;
    } else {
      drift = 5 - remainder;
      roundUp++;
    }
    if (value < 0 && drift != 0) {
      drift = -drift;
    }
    driftCounts[drift] = (driftCounts[drift] ?? 0) + 1;
  }
  int repeatedDrifts = 0;
  for (final count in driftCounts.values) {
    if (count > 1) {
      repeatedDrifts++;
    }
  }
  return [roundDown, exact, roundUp, repeatedDrifts];
}

@pragma('vm:entry-point')
void main() {
  assert(classifyNickelRoundingBias([]).toString() == '[0, 0, 0, 0]');
  assert(classifyNickelRoundingBias([1, 2, 3, 4]).toString() == '[2, 0, 2, 0]');
  assert(classifyNickelRoundingBias([0, 1, 5, 6, 9]).toString() == '[2, 2, 1, 2]');
  print('All tests passed!');
}