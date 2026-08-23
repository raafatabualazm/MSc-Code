@pragma('vm:entry-point')
int tallyAsciiUndoInk(List<String> ops, int targetWidth) {
  List<String> rows = [];
  List<String> history = [];
  for (final op in ops) {
    if (op.startsWith('+')) {
      String row = op.substring(1);
      if (row.length <= targetWidth) {
        rows.add(row);
        history.add('+');
      } else {
        history.add('x');
      }
    } else if (op == '-') {
      history.add(rows.isNotEmpty ? rows.removeLast() : '');
    } else if (op == '!' && history.isNotEmpty) {
      String last = history.removeLast();
      if (last == '+') {
        rows.removeLast();
      } else if (last.isNotEmpty && last != 'x') {
        rows.add(last);
      }
    }
  }
  int ink = 0;
  for (final row in rows) {
    for (int i = 0; i < row.length; i++) {
      if (row[i] != ' ') ink++;
    }
  }
  return ink;
}

@pragma('vm:entry-point')
void main() {
  assert(tallyAsciiUndoInk([], 4) == 0);
  assert(tallyAsciiUndoInk(['+##', '-', '!'], 5) == 2);
  assert(tallyAsciiUndoInk(['+----'], 3) == 0);
  print('All tests passed!');
}