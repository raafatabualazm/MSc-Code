@pragma('vm:entry-point')
int deckRevealUndoScore(List<String> actions) {
  List<int> pile = [];
  List<String> history = [];
  int score = 0;
  for (String op in actions) {
    if (op.startsWith('D')) {
      pile.add(int.parse(op.substring(1)));
      history.add('D');
    } else if (op == 'R') {
      if (pile.isNotEmpty) {
        int v = pile.removeLast();
        score += v;
        history.add('R$v');
      } else {
        score -= 2;
      }
    } else if (op == 'B') {
      if (pile.isNotEmpty) {
        history.add('B${pile.removeLast()}');
      }
    } else if (op == 'U' && history.isNotEmpty) {
      String last = history.removeLast();
      if (last == 'D') {
        pile.removeLast();
      } else {
        int v = int.parse(last.substring(1));
        pile.add(v);
        if (last.startsWith('R')) score -= v;
      }
    }
  }
  return score + pile.length;
}

@pragma('vm:entry-point')
void main() {
  assert(deckRevealUndoScore([]) == 0);
  assert(deckRevealUndoScore(['D5', 'R']) == 5);
  assert(deckRevealUndoScore(['D4', 'R', 'U']) == 1);
  print('All tests passed!');
}