@pragma('vm:entry-point')
int countCardHandPoints(String hand) {
  if (hand.isEmpty) return 0;
  final tokens = hand.split(' ');
  const ranks = '23456789TJQKA';
  const suits = 'cdhs';
  int total = 0;
  for (final token in tokens) {
    if (token.length != 2) continue;
    final rank = token[0];
    final suit = token[1];
    if (!ranks.contains(rank) || !suits.contains(suit)) continue;
    if (rank == 'A') {
      total += 11;
    } else if ('TJQK'.contains(rank)) {
      total += 10;
    } else {
      total += int.parse(rank);
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(countCardHandPoints('') == 0);
  assert(countCardHandPoints('Ah 2c 3d') == 16);
  assert(countCardHandPoints('Th Jd Qs Kc') == 40);
  print('All tests passed!');
}