@pragma('vm:entry-point')
List<int> processLedgerUndoRedo(List<int> operations) {
  var balances = <int>[], undo = <int>[], redo = <int>[];
  int balance = 0;
  for (var op in operations) {
    if (op == -1) {
      if (undo.isNotEmpty) { int last = undo.removeLast(); redo.add(last); balance -= last; }
    } else if (op == -2) {
      if (redo.isNotEmpty) { int last = redo.removeLast(); undo.add(last); balance += last; }
    } else {
      undo.add(op); balance += op; redo.clear();
    }
    balances.add(balance);
  }
  return balances;
}

@pragma('vm:entry-point')
void main() {
  assert(processLedgerUndoRedo([]).isEmpty);
  assert(processLedgerUndoRedo([10, -1]).toString() == '[10, 0]');
  assert(processLedgerUndoRedo([10, -1, -2]).toString() == '[10, 0, 10]');
  print('All tests passed!');
}