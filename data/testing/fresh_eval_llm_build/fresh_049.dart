@pragma('vm:entry-point')
List<String> tokenizeCardTradeLedger(String notes) {
  const ranks = 'A23456789TJQK';
  const suits = 'SHDC';
  List<String> result = [];
  String previous = '';
  for (String part in notes.split(',')) {
    String token = part.trim().toUpperCase();
    if (token.isEmpty) continue;
    bool marked = token.endsWith('!');
    if (marked) token = token.substring(0, token.length - 1);
    if (token.length == 2 && ranks.contains(token[0]) && suits.contains(token[1])) {
      if (token == previous) {
        result.add('DOUBLE');
      } else if (!marked) {
        result.add(token);
      }
      previous = token;
    } else if (token == 'JOKER' && !marked) {
      result.add('WILD');
      previous = '';
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(tokenizeCardTradeLedger('').isEmpty);
  assert(tokenizeCardTradeLedger('AS, AS').toString() == '[AS, DOUBLE]');
  assert(tokenizeCardTradeLedger('joker!,qh').length == 1);
  print('All tests passed!');
}