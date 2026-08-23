@pragma('vm:entry-point')
int countDeckOrderPenalties(List<String> cards, String suitOrder) {
  const ranks = '23456789TJQKA';
  int score = 0;
  for (int i = 0; i < cards.length - 1; i++) {
    String a = cards[i];
    String b = cards[i + 1];
    int suitA = suitOrder.indexOf(a[a.length - 1]);
    int suitB = suitOrder.indexOf(b[b.length - 1]);
    int rankA = ranks.indexOf(a[0]);
    int rankB = ranks.indexOf(b[0]);
    if (suitA > suitB) {
      score += 3;
    } else if (suitA == suitB) {
      if (rankA > rankB) {
        score += 2;
      } else if (rankB - rankA > 2) {
        score += 1;
      }
    } else if (rankA == rankB) {
      score += 1;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(countDeckOrderPenalties([], 'SHDC') == 0);
  assert(countDeckOrderPenalties(['2S', '5S'], 'SHDC') == 1);
  assert(countDeckOrderPenalties(['5S', '2S'], 'SHDC') == 2);
  print('All tests passed!');
}