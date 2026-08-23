@pragma('vm:entry-point')
int asciiRowRunChecksum(String sketch) {
  int score = 0, row = 0, col = 0;
  String last = '';
  void paint(String ch) {
    if (ch == '#') {
      score += row + col + 1;
    } else if (ch == '*') {
      score += col.isEven ? 2 : 1;
    } else if (ch == '.' && col > 0 && last == '.') {
      score -= 1;
    }
    if (ch == last && ch != '.') score += 1;
    last = ch;
    col++;
  }
  for (int i = 0; i < sketch.length; i++) {
    String ch = sketch[i];
    if (ch == '/') {
      if (col.isOdd) score += 3;
      row++;
      col = 0;
      last = '';
    } else if (ch.codeUnitAt(0) >= 50 && ch.codeUnitAt(0) <= 57 && i + 1 < sketch.length) {
      int count = ch.codeUnitAt(0) - 48;
      for (int j = 0; j < count; j++) {
        paint(sketch[i + 1]);
      }
      i++;
    } else {
      paint(ch);
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(asciiRowRunChecksum('') == 0);
  assert(asciiRowRunChecksum('##') == 4);
  assert(asciiRowRunChecksum('3*/2#') == 16);
  print('All tests passed!');
}