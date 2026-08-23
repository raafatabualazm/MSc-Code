@pragma('vm:entry-point')
List<int> streakOfSameRank(List<int> deck) {
  if (deck.isEmpty) return [];
  int start = 0, bestStart = 0, bestLen = 1;
  for (int i = 1; i < deck.length; i++) {
    if (deck[i] == deck[i - 1]) {
      int len = i - start + 1;
      if (len > bestLen) {
        bestLen = len;
        bestStart = start;
      }
    } else {
      start = i;
    }
  }
  return [bestStart, bestStart + bestLen - 1];
}

@pragma('vm:entry-point')
void main() {
  assert(streakOfSameRank([]).toString() == '[]');
  assert(streakOfSameRank([5]).toString() == '[0, 0]');
  assert(streakOfSameRank([4, 4, 3, 3]).toString() == '[0, 1]');
  print('All tests passed!');
}