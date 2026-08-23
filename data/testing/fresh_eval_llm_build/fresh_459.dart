@pragma('vm:entry-point')
String mapTideHarborWindows(List<int> readings) {
  if (readings.isEmpty) return 'empty';
  int n = readings.length;
  List<List<bool>> dp = List.generate(n, (_) => List.filled(n, false));
  int bestLen = 0;
  int count = 0;
  for (int len = 2; len <= n; len++) {
    for (int start = 0; start + len <= n; start++) {
      int end = start + len - 1;
      bool smooth = true;
      bool hasFlat = false;
      for (int k = start + 1; k <= end; k++) {
        int diff = (readings[k] - readings[k - 1]).abs();
        if (diff > 2) {
          smooth = false;
          break;
        } else if (diff == 0) {
          hasFlat = true;
        }
      }
      if (!smooth || !hasFlat) continue;
      int edge = (readings[start] - readings[end]).abs();
      if (edge > 1) continue;
      if (len > 3 && !dp[start + 1][end - 1] && readings[start] != readings[end]) {
        continue;
      }
      dp[start][end] = true;
      if (len > bestLen) {
        bestLen = len;
        count = 1;
      } else if (len == bestLen) {
        count++;
      }
    }
  }
  return bestLen == 0 ? 'none' : '$bestLen:$count';
}

@pragma('vm:entry-point')
void main() {
  assert(mapTideHarborWindows([]) == 'empty');
  assert(mapTideHarborWindows([1, 1, 2]) == '3:1');
  assert(mapTideHarborWindows([3, 3, 4, 3]) == '4:1');
  print('All tests passed!');
}