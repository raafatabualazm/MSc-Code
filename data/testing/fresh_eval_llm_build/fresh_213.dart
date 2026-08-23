@pragma('vm:entry-point')
int scoreDiceRerollLedger(List<String> rounds) {
  final ledger = <int>[];
  int total = 0;
  for (final round in rounds) {
    if (round == 'erase') {
      if (ledger.isNotEmpty) total -= ledger.removeLast();
    } else {
      final value = round == 'split' ? (ledger.isEmpty ? 0 : ledger.last ~/ 2) : int.parse(round);
      ledger.add(value);
      total += value;
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(scoreDiceRerollLedger([]) == 0);
  assert(scoreDiceRerollLedger(['5', 'split']) == 7);
  assert(scoreDiceRerollLedger(['3', '4', 'erase', 'split']) == 4);
  print('All tests passed!');
}