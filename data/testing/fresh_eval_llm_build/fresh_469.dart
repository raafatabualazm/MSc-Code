@pragma('vm:entry-point')
int tallyChessSquarePattern(String text) {
  List<String> parts = [];
  String current = '';
  for (int i = 0; i < text.length; i++) {
    String c = text[i].toLowerCase();
    int code = c.codeUnitAt(0);
    bool alphaNum = (code >= 97 && code <= 122) || (code >= 48 && code <= 57);
    if (alphaNum) {
      current += c;
    } else if (current.isNotEmpty) {
      parts.add(current);
      current = '';
    }
  }
  if (current.isNotEmpty) parts.add(current);
  int score = 0;
  String prev = '';
  for (String p in parts) {
    bool valid = p.length == 2 && p.codeUnitAt(0) >= 97 && p.codeUnitAt(0) <= 104 && p.codeUnitAt(1) >= 49 && p.codeUnitAt(1) <= 56;
    if (!valid) {
      score -= 2;
      continue;
    }
    if (prev.isEmpty) {
      score += 1;
    } else if (p == prev) {
      score += 4;
    } else {
      int color = ((p.codeUnitAt(0) - 97) + (p.codeUnitAt(1) - 49)) & 1;
      int prevColor = ((prev.codeUnitAt(0) - 97) + (prev.codeUnitAt(1) - 49)) & 1;
      score += color == prevColor ? 2 : -1;
    }
    prev = p;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(tallyChessSquarePattern('a1 b2') == 3);
  assert(tallyChessSquarePattern('a1,b1') == 0);
  assert(tallyChessSquarePattern('z9') == -2);
  print('All tests passed!');
}