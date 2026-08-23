@pragma('vm:entry-point')
List<int> locateBalancedDeckWindow(List<int> deck) {
  if (deck.isEmpty) return [];
  final Map<int, int> counts = {};
  int left = 0;
  int bestLen = 0;
  int bestSum = -1;
  int bestL = -1;
  int bestR = -1;
  int sum = 0;
  for (int right = 0; right < deck.length; right++) {
    final int card = deck[right];
    if (card <= 0) {
      counts.clear();
      sum = 0;
      left = right + 1;
      continue;
    }
    if (right > left && deck[right] == deck[right - 1]) {
      while (left < right) {
        counts[deck[left]] = (counts[deck[left]] ?? 1) - 1;
        sum -= deck[left];
        if (counts[deck[left]] == 0) counts.remove(deck[left]);
        left++;
      }
    }
    counts[card] = (counts[card] ?? 0) + 1;
    sum += card;
    while ((counts[card] ?? 0) > 2) {
      counts[deck[left]] = (counts[deck[left]] ?? 1) - 1;
      sum -= deck[left];
      if (counts[deck[left]] == 0) counts.remove(deck[left]);
      left++;
    }
    final int len = right - left + 1;
    if (len > bestLen || (len == bestLen && sum > bestSum)) {
      bestLen = len;
      bestSum = sum;
      bestL = left;
      bestR = right;
    }
  }
  if (bestLen == 0) return [];
  return [bestL, bestR, bestSum];
}

@pragma('vm:entry-point')
void main() {
  assert(locateBalancedDeckWindow([]).toString() == '[]');
  assert(locateBalancedDeckWindow([1, 2, 1, 2]).toString() == '[0, 3, 6]');
  assert(locateBalancedDeckWindow([4, 5, -1, 6, 7]).toString() == '[3, 4, 13]');
  print('All tests passed!');
}