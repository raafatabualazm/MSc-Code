@pragma('vm:entry-point')
int coveredCardLanePoints(List<int> deckCenters, int reach) {
  if (reach < 0 || deckCenters.isEmpty) {
    return 0;
  }
  int minX = 1 << 30;
  int maxX = -(1 << 30);
  for (int i = 0; i < deckCenters.length; i += 2) {
    int y = deckCenters[i + 1].abs();
    if (y <= reach) {
      int span = reach - y;
      int left = deckCenters[i] - span;
      int right = deckCenters[i] + span;
      if (left < minX) minX = left;
      if (right > maxX) maxX = right;
    }
  }
  if (minX > maxX) return 0;
  int covered = 0;
  for (int x = minX; x <= maxX; x++) {
    for (int i = 0; i < deckCenters.length; i += 2) {
      if ((x - deckCenters[i]).abs() + deckCenters[i + 1].abs() <= reach) {
        covered++;
        break;
      }
    }
  }
  return covered;
}

@pragma('vm:entry-point')
void main() {
  assert(coveredCardLanePoints([], 3) == 0);
  assert(coveredCardLanePoints([0, 0, 2, 0], 1) == 5);
  assert(coveredCardLanePoints([0, 2], 2) == 1);
  print('All tests passed!');
}