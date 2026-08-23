@pragma('vm:entry-point')
double layeredAsciiEchoScore(List<String> rows) {
  if (rows.isEmpty) return 0.0;
  int width = 0;
  for (final row in rows) {
    if (row.length > width) width = row.length;
  }
  List<int> heights = List.filled(width, 0);
  double score = 0.0;
  String prev = '';
  for (final row in rows) {
    List<int> next = List.filled(width, 0);
    for (int c = 0; c < width; c++) {
      String ch = c < row.length ? row[c] : ' ';
      if (ch != ' ') {
        next[c] = (c < prev.length && prev[c] == ch) ? heights[c] + 1 : 1;
        score += next[c] >= 3 ? 2.0 : (next[c] == 2 ? 1.5 : 0.5);
      } else if (c > 0 && next[c - 1] >= 2) {
        score -= 0.5;
      }
    }
    heights = next;
    prev = row;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(layeredAsciiEchoScore([]) == 0.0);
  assert(layeredAsciiEchoScore(['*', '*']) == 2.0);
  assert(layeredAsciiEchoScore(['@@', '@@', '@ ']) == 5.5);
  print('All tests passed!');
}