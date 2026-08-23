@pragma('vm:entry-point')
String replayAsciiRowUndo(String script, int width) {
  List<String> history = [];
  List<String> rows = [];
  String row = '';
  for (int i = 0; i < script.length; i++) {
    String c = script[i];
    if (c == '#' || c == '.') {
      if (row.length < width) row += c;
    } else if (c == '<') {
      if (row.isNotEmpty) row = row.substring(0, row.length - 1);
    } else if (c == '(') {
      history.add(row);
    } else if (c == ')') {
      if (history.isNotEmpty) row = history.removeLast();
    } else if (c == '|') {
      rows.add(row.length == width ? row : '${row}!');
      row = '';
    }
  }
  if (row.isNotEmpty) rows.add(row.length == width ? row : '${row}!');
  return rows.join('/');
}

@pragma('vm:entry-point')
void main() {
  assert(replayAsciiRowUndo("#.|", 2) == "#.");
  assert(replayAsciiRowUndo("##|(.)|", 2) == "##/!");
  assert(replayAsciiRowUndo("#<", 2) == "");
  print('All tests passed!');
}