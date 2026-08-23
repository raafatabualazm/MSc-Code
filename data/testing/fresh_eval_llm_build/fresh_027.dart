@pragma('vm:entry-point')
String rewindMazeCellRoute(String script, int row, int col) {
  List<String> trail = [];
  for (int i = 0; i < script.length; i++) {
    String c = script[i];
    if ('NSEW'.contains(c)) {
      trail.add('$row,$col');
      if (c == 'N') row--; else if (c == 'S') row++; else if (c == 'E') col++; else col--;
    } else if (c == 'U') {
      if (trail.isNotEmpty) {
        List<String> p = trail.removeLast().split(',');
        row = int.parse(p[0]);
        col = int.parse(p[1]);
      }
    } else if (c == 'B' && trail.length >= 2) {
      trail.removeLast();
      List<String> p = trail.removeLast().split(',');
      row = int.parse(p[0]);
      col = int.parse(p[1]);
    }
  }
  return '$row:$col:${trail.length}';
}

@pragma('vm:entry-point')
void main() {
  assert(rewindMazeCellRoute('', 0, 0) == '0:0:0');
  assert(rewindMazeCellRoute('NU', 2, 3) == '2:3:0');
  assert(rewindMazeCellRoute('NESWB', 0, 0) == '-1:1:2');
  print('All tests passed!');
}