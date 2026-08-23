@pragma('vm:entry-point')
List<int> suitFilteredCardValues(String hand, String suit) {
  if (hand.isEmpty) return [];
  final faceMap = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'T':10,'J':11,'Q':12,'K':13,'A':14};
  final List<int> result = [];
  for (final token in hand.split(' ')) {
    if (token.length == 2 && token[1] == suit) {
      final v = faceMap[token[0]];
      if (v != null) result.add(v);
    }
  }
  result.sort();
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(suitFilteredCardValues('2H 5H KH', 'H').toString() == '[2, 5, 13]');
  assert(suitFilteredCardValues('AS KS QS', 'S').toString() == '[12, 13, 14]');
  assert(suitFilteredCardValues('', 'H').toString() == '[]');
  print('All tests passed!');
}