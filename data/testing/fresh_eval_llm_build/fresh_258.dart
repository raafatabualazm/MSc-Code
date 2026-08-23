@pragma('vm:entry-point')
String tideSlackWindow(List<int> heights) {
  if (heights.length < 2) return 'none';
  int bestStart = -1, bestEnd = -1, bestTurns = -1;
  int start = 0, turns = 0, lastDir = 0;
  for (int i = 1; i < heights.length; i++) {
    int diff = heights[i] - heights[i - 1];
    if (diff.abs() <= 2) {
      int dir = diff == 0 ? lastDir : (diff > 0 ? 1 : -1);
      if (lastDir != 0 && dir != lastDir) turns++;
      lastDir = dir;
      int span = i - start, bestSpan = bestEnd - bestStart;
      if (span > bestSpan || (span == bestSpan && turns > bestTurns)) {
        bestStart = start;
        bestEnd = i;
        bestTurns = turns;
      }
    } else {
      start = i;
      turns = 0;
      lastDir = 0;
    }
  }
  return (bestStart == -1 || bestEnd - bestStart < 2) ? 'none' : '$bestStart-$bestEnd:$bestTurns';
}

@pragma('vm:entry-point')
void main() {
  assert(tideSlackWindow([1, 2, 3]) == '0-2:0');
  assert(tideSlackWindow([1, 4, 1]) == 'none');
  assert(tideSlackWindow([0, 1, 0, 1]) == '0-3:2');
  print('All tests passed!');
}