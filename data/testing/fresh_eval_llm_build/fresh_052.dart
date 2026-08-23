@pragma('vm:entry-point')
List<num> reconcileLedgerBands(List<List<int>> ledger) {
  List<num> result = [];
  for (final row in ledger) {
    num score = 0;
    for (int i = 0; i < row.length; i++) {
      int entry = row[i];
      if (entry < 0) {
        score += -entry;
        if (i > 0 && row[i - 1] > 0) score -= 0.5;
      } else if (entry > 0) {
        score += entry / 2;
        if (i + 1 < row.length && row[i + 1] < 0) score += 1;
      } else {
        score += (i > 0 && row[i - 1] == 0) ? 0.25 : -0.25;
      }
    }
    result.add(score);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(reconcileLedgerBands([]).toString() == '[]');
  assert(reconcileLedgerBands([[5, -2]]).toString() == '[5.0]');
  assert(reconcileLedgerBands([[0, 0], [-1, 2, -3]]).toString() == '[0.0, 5.5]');
  print('All tests passed!');
}