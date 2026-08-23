@pragma('vm:entry-point')
List<int> findLongestStraightCorridor(List<int> masks) {
  int n = masks.length, start = 0, maxL = 0, bestS = -1, bestE = -1, dir = -1;
  for (int i = 0; i < n; i++) {
    int m = masks[i];
    if (m != 10 && m != 5) { start = i + 1; dir = -1; continue; }
    if (dir == -1) dir = m;
    else if (m != dir) { start = i; dir = m; }
    int len = i - start + 1;
    if (len > maxL) { maxL = len; bestS = start; bestE = i; }
  }
  return bestS == -1 ? [] : [bestS, bestE];
}

@pragma('vm:entry-point')
void main() {
  assert(findLongestStraightCorridor([10,10,5,5,5]).toString() == '[2, 4]');
  assert(findLongestStraightCorridor([]).toString() == '[]');
  assert(findLongestStraightCorridor([10,10,10]).toString() == '[0, 2]');
  print('All tests passed!');
}