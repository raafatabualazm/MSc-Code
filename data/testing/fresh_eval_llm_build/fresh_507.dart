@pragma('vm:entry-point')
List<int> auditPasswordExpiryPatterns(List<int> days) {
  int denseRuns = 0;
  int forcedResets = 0;
  int echoPairs = 0;
  if (days.isEmpty) return [0, 0, 0];
  for (int i = 0; i < days.length; i++) {
    int start = days[i];
    if (start < 0) continue;
    int run = 1;
    for (int j = i + 1; j < days.length; j++) {
      if (days[j] < 0) break;
      int gap = days[j] - days[j - 1];
      if (gap < 0) {
        forcedResets++;
        break;
      }
      if (gap == 0) {
        echoPairs++;
        continue;
      }
      if (gap <= 2) {
        run++;
      } else {
        if (gap > 45) forcedResets++;
        break;
      }
    }
    if (run >= 3) denseRuns++;
    for (int k = i + 1; k < days.length; k++) {
      int span = days[k] - start;
      if (span < 0) continue;
      if (span > 60) break;
      if (span >= 7 && span <= 14 && ((days[k] + start) & 1) == 0) {
        echoPairs++;
      }
    }
  }
  return [denseRuns, forcedResets, echoPairs];
}

@pragma('vm:entry-point')
void main() {
  assert(auditPasswordExpiryPatterns([]).toString() == '[0, 0, 0]');
  assert(auditPasswordExpiryPatterns([1, 2, 3]).toString() == '[1, 0, 0]');
  assert(auditPasswordExpiryPatterns([2, 4, 6, 14]).toString() == '[1, 0, 3]');
  print('All tests passed!');
}