@pragma('vm:entry-point')
String reorderAsciiStrips(List<String> rows, int streakLimit) {
  if (rows.isEmpty) return '';
  List<String> filtered = [];
  for (final row in rows) {
    if (row.isEmpty) continue;
    filtered.add(row);
  }
  if (filtered.isEmpty) return '';
  int scoreRow(String row) {
    int score = 0;
    for (int i = 0; i < row.length; i++) {
      String c = row[i];
      if (c == '#') {
        score += 3;
      } else if (c == '/' || c == '\\') {
        score += 2;
      } else if (c == '_') {
        score += 1;
      } else if (c == '.') {
        score -= 1;
      }
      for (int j = i + 1; j < row.length && row[j] == c; j++) {
        score++;
        if (j - i >= streakLimit.abs()) break;
      }
    }
    if (row.length < streakLimit) {
      score -= 2;
    } else if (row.length == streakLimit) {
      score += 1;
    }
    return score;
  }

  filtered.sort((a, b) {
    int sa = scoreRow(a);
    int sb = scoreRow(b);
    if (sa != sb) return sb - sa;
    if (a.length != b.length) return a.length - b.length;
    return a.compareTo(b);
  });
  return filtered.join('|');
}

@pragma('vm:entry-point')
void main() {
  assert(reorderAsciiStrips(['#', '##'], 2) == '##|#');
  assert(reorderAsciiStrips(['', ''], 3) == '');
  assert(reorderAsciiStrips(['/\\', '__', '..'], 2) == '/\\|__|..');
  print('All tests passed!');
}