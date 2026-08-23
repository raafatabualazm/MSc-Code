@pragma('vm:entry-point')
String summarizeDeckTokens(String log) {
  const ranks = 'A23456789TJQK';
  const suits = 'SHDC';
  var seen = <String>{};
  var valid = 0, dup = 0, invalid = 0;
  for (var raw in log.split(RegExp(r'[ ,|]+'))) {
    if (raw.isEmpty) continue;
    var card = raw.toUpperCase();
    if (card.length == 2 &&
        ranks.contains(card[0]) &&
        suits.contains(card[1])) {
      if (seen.add(card)) {
        valid++;
      } else {
        dup++;
      }
    } else {
      invalid++;
    }
  }
  if (valid == 52 && dup == 0 && invalid == 0) return 'complete';
  if (invalid > 0) return 'chaos:$valid/$dup/$invalid';
  if (dup > 0) return 'stacked:$valid/$dup';
  return valid == 0 ? 'empty' : 'clean:$valid';
}

@pragma('vm:entry-point')
void main() {
  assert(summarizeDeckTokens('AS KD') == 'clean:2');
  assert(summarizeDeckTokens('AS AS') == 'stacked:1/1');
  assert(summarizeDeckTokens('') == 'empty');
  print('All tests passed!');
}