@pragma('vm:entry-point')
int simulateChessUndoRedo(List<String> ops) {
  var undoStack = <int>[];
  var redoStack = <int>[];
  var counts = <int, int>{};
  for (var op in ops) {
    if (op == 'undo') {
      if (undoStack.isNotEmpty) {
        int sq = undoStack.removeLast();
        redoStack.add(sq);
        counts[sq] = (counts[sq] ?? 0) - 1;
      }
    } else if (op == 'redo') {
      if (redoStack.isNotEmpty) {
        int sq = redoStack.removeLast();
        undoStack.add(sq);
        counts[sq] = (counts[sq] ?? 0) + 1;
      }
    } else if (op == 'clear') {
      undoStack.clear();
      redoStack.clear();
      counts.clear();
    } else {
      int file = op.codeUnitAt(0) - 97;
      int rank = int.parse(op[1]) - 1;
      int sq = file * 8 + rank;
      undoStack.add(sq);
      counts[sq] = (counts[sq] ?? 0) + 1;
      redoStack.clear();
    }
  }
  return counts.values.where((c) => c > 0).length;
}

@pragma('vm:entry-point')
void main() {
  assert(simulateChessUndoRedo([]) == 0);
  assert(simulateChessUndoRedo(['e4', 'undo', 'redo']) == 1);
  assert(simulateChessUndoRedo(['a1', 'h8', 'a1', 'undo']) == 2);
  print('All tests passed!');
}