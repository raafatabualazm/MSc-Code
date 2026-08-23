@pragma('vm:entry-point')
int scoreDeckCascadeOrder(List<String> cards, bool blackFirst) {
  if (cards.isEmpty) return 0;
  final deck = List<String>.from(cards);
  final suitOrder = blackFirst
      ? {'S': 0, 'C': 1, 'H': 2, 'D': 3}
      : {'H': 0, 'D': 1, 'S': 2, 'C': 3};
  int rankOf(String card) {
    final face = card.substring(0, card.length - 1);
    if (face == 'A') return 1;
    if (face == 'J') return 11;
    if (face == 'Q') return 12;
    if (face == 'K') return 13;
    return int.parse(face);
  }
  deck.sort((a, b) {
    final sa = a.substring(a.length - 1);
    final sb = b.substring(b.length - 1);
    final diff = suitOrder[sa]! - suitOrder[sb]!;
    return diff != 0 ? diff : rankOf(a) - rankOf(b);
  });
  int score = 0;
  for (int i = 0; i < deck.length; i++) {
    final ri = rankOf(deck[i]);
    final si = deck[i].substring(deck[i].length - 1);
    bool linked = false;
    for (int j = i + 1; j < deck.length && j <= i + 3; j++) {
      final rj = rankOf(deck[j]);
      final sj = deck[j].substring(deck[j].length - 1);
      if (si == sj && rj == ri + 1) {
        score += blackFirst ? 3 : 2;
        linked = true;
        break;
      }
      if (si == sj && rj == ri) {
        score -= 1;
        continue;
      }
      if (suitOrder[sj]! < suitOrder[si]! && rj > ri) {
        score += 2;
      } else if ((rj - ri).abs() > 5) {
        score += 1;
      }
    }
    if (!linked) {
      if (ri >= 11) {
        score += 2;
      } else if (ri <= 3) {
        score -= 1;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(scoreDeckCascadeOrder([], true) == 0);
  assert(scoreDeckCascadeOrder(['2S', '3S'], true) == 2);
  assert(scoreDeckCascadeOrder(['QH', 'KH'], false) == 4);
  print('All tests passed!');
}