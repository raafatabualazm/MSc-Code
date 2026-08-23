@pragma('vm:entry-point')
List<int> settleCardDeckLog(List<String> actions) {
  List<int> deck = [];
  List<List<int>> history = [];
  for (var action in actions) {
    if (action == 'U') {
      if (history.isNotEmpty) {
        deck = List<int>.from(history.removeLast());
      }
    } else {
      history.add(List<int>.from(deck));
      if (action == 'R') {
        if (deck.isNotEmpty) {
          deck.removeLast();
        }
      } else if (action == 'S') {
        if (deck.length > 1) {
          int top = deck.removeLast();
          int below = deck.removeLast();
          deck.add(top > below ? top - below : top + below);
        }
      } else {
        int card = int.parse(action.substring(1));
        deck.add(card.isEven ? card ~/ 2 : card);
      }
    }
  }
  return deck;
}

@pragma('vm:entry-point')
void main() {
  assert(settleCardDeckLog([]).toString() == '[]');
  assert(settleCardDeckLog(['P8', 'R', 'U']).toString() == '[4]');
  assert(settleCardDeckLog(['P5', 'P3', 'S']).toString() == '[8]');
  print('All tests passed!');
}