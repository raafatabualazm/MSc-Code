@pragma('vm:entry-point')
List<String> classifyChessSquareTransitions(String text) {
  List<String> out = [];
  String token = '';
  String? prev;
  for (int i = 0; i <= text.length; i++) {
    String ch = i < text.length ? text[i] : ' ';
    if ((ch.codeUnitAt(0) >= 48 && ch.codeUnitAt(0) <= 57) || (ch.toLowerCase().codeUnitAt(0) >= 97 && ch.toLowerCase().codeUnitAt(0) <= 122)) {
      token += ch;
    } else {
      if (token.length == 2) {
        String sq = token.toLowerCase();
        int f = sq.codeUnitAt(0) - 97, r = sq.codeUnitAt(1) - 49;
        if (f >= 0 && f < 8 && r >= 0 && r < 8) {
          if (prev != null) {
            int dx = (f - (prev.codeUnitAt(0) - 97)).abs(), dy = (r - (prev.codeUnitAt(1) - 49)).abs();
            if ((dx == 1 && dy == 2) || (dx == 2 && dy == 1)) out.add(sq); else if (dx == 0 || dy == 0) out.add('$prev:$sq'); else if (dx == 1 && dy == 1) out.add('${sq}*');
          }
          prev = sq;
        }
      }
      token = '';
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(classifyChessSquareTransitions('a1 b3').toString() == '[b3]');
  assert(classifyChessSquareTransitions('c3 d4').toString() == '[d4*]');
  assert(classifyChessSquareTransitions('aa1 b2').length == 0);
  print('All tests passed!');
}