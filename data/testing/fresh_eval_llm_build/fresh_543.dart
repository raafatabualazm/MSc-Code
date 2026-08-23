@pragma('vm:entry-point')
List<double> shelfCodeIntervalLedger(List<int> shelfCodes) {
  if (shelfCodes.isEmpty) {
    return [0.0, 0.0, 0.0];
  }
  double aligned = 0.0;
  double repeats = 0.0;
  double blocked = 0.0;
  for (int i = 0; i < shelfCodes.length; i++) {
    int current = shelfCodes[i];
    if (current == 0) {
      blocked += 0.5;
      continue;
    }
    int day = current.abs();
    if (day % 10 == 0) {
      aligned += 1.0;
    } else if (day % 5 == 0) {
      aligned += 0.5;
    }
    for (int j = 0; j < i; j++) {
      int earlier = shelfCodes[j].abs();
      int gap = day - earlier;
      if (gap <= 0) {
        continue;
      }
      if (gap < 3) {
        repeats += 1.0;
      } else if (gap == 7) {
        aligned += 2.0;
      } else if (gap % 7 == 0) {
        aligned += 0.5;
      }
      if ((current < 0) != (shelfCodes[j] < 0) && day % 2 == earlier % 2) {
        blocked += 0.5;
      }
    }
  }
  return [aligned, repeats, blocked];
}

@pragma('vm:entry-point')
void main() {
  assert(shelfCodeIntervalLedger([]).toString() == '[0.0, 0.0, 0.0]');
  assert(shelfCodeIntervalLedger([1, 8]).toString() == '[2.0, 0.0, 0.0]');
  assert(shelfCodeIntervalLedger([-2, 4]).toString() == '[0.0, 1.0, 0.5]');
  print('All tests passed!');
}