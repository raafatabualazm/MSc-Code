@pragma('vm:entry-point')
List<int> collectMazeCellMarks(String path, int limit) {
  int pos = 0;
  int repeat = 0;
  bool blocked = false;
  List<int> marks = [];
  for (int i = 0; i < path.length; i++) {
    String c = path[i];
    if (c == '#') {
      blocked = !blocked;
    } else if (c.codeUnitAt(0) >= 48 && c.codeUnitAt(0) <= 57) {
      repeat = repeat * 10 + (c.codeUnitAt(0) - 48);
    } else {
      int steps = repeat == 0 ? 1 : repeat;
      repeat = 0;
      if (!blocked && (c == 'R' || c == 'L')) {
        pos += c == 'R' ? steps : -steps;
      } else if (c == 'M' && pos.abs() <= limit) {
        marks.add(pos);
      }
    }
  }
  return marks;
}

@pragma('vm:entry-point')
void main() {
  assert(collectMazeCellMarks('RMLM', 5).toString() == '[1, 0]');
  assert(collectMazeCellMarks('#R#MRM', 5).length == 2);
  assert(collectMazeCellMarks('12RM', 11).toString() == '[]');
  print('All tests passed!');
}