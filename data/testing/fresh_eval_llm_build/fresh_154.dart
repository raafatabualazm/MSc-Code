@pragma('vm:entry-point')
List<int> balancedDiscardRanks(List<String> cards, int copies) {
  var counts = <int, int>{};
  var suitMasks = <int, int>{};
  for (var card in cards) {
    var rank = 'A23456789TJQK'.indexOf(card[0]) + 1;
    var suit = 'CDHS'.indexOf(card[card.length - 1]);
    counts[rank] = (counts[rank] ?? 0) + 1;
    suitMasks[rank] = (suitMasks[rank] ?? 0) | (1 << suit);
  }
  return counts.keys.where((r) => counts[r] == copies && (suitMasks[r]! & (suitMasks[r]! - 1)) != 0).toList()..sort();
}

@pragma('vm:entry-point')
void main() {
  assert(balancedDiscardRanks(['AH', 'AD'], 2).toString() == '[1]');
  assert(balancedDiscardRanks([], 1).toString() == '[]');
  assert(balancedDiscardRanks(['KH', 'KD', 'AC', 'AS'], 2).length == 2);
  print('All tests passed!');
}