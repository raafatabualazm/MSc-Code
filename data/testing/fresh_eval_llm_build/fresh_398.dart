@pragma('vm:entry-point')
Map<String, int> bracketRestWindows(List<int> matchDays, int targetGap) {
  int compressed = 0, balanced = 0, extended = 0, totalDays = 0;
  for (int i = 1; i < matchDays.length; i++) {
    int gap = matchDays[i] - matchDays[i - 1];
    totalDays += gap;
    if (gap < targetGap) {
      compressed++;
    } else if (gap > targetGap) {
      extended++;
    } else {
      balanced++;
    }
  }
  return {
    'compressed': compressed,
    'balanced': balanced,
    'extended': extended,
    'totalDays': totalDays
  };
}

@pragma('vm:entry-point')
void main() {
  assert(bracketRestWindows([], 3).toString() == '{compressed: 0, balanced: 0, extended: 0, totalDays: 0}');
  assert(bracketRestWindows([1, 3, 5], 2).toString() == '{compressed: 0, balanced: 2, extended: 0, totalDays: 4}');
  assert(bracketRestWindows([1, 2, 4, 8], 2).toString() == '{compressed: 1, balanced: 1, extended: 1, totalDays: 7}');
  print('All tests passed!');
}