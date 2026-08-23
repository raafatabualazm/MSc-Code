@pragma('vm:entry-point')
int evaluatePassWindowDrift(List<int> windows) {
  int score = 0;
  for (int i = 0; i + 1 < windows.length; i += 2) {
    int start = windows[i];
    int end = windows[i + 1];
    if (end < start) {
      score -= (start - end) * 3;
    } else {
      int span = end - start;
      score += span;
      if (i >= 2) {
        int gap = start - windows[i - 1];
        if (gap == 0) score += 2;
        else if (gap < 0) score -= gap.abs() * 2;
        else if (gap > 3) score -= gap - 3;
      }
      if (span > 5) score += 1;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(evaluatePassWindowDrift([]) == 0);
  assert(evaluatePassWindowDrift([1, 4, 4, 6]) == 7);
  assert(evaluatePassWindowDrift([0, 6, 6, 12, 10, 8]) == 10);
  print('All tests passed!');
}