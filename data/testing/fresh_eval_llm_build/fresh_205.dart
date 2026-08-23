@pragma('vm:entry-point')
int scoreBracketTape(String tape) {
  if (tape.isEmpty) return 0;
  int depth = 0, score = 0, i = 0;
  bool needTeam = true, sawArrow = false;
  String left = '';
  while (i < tape.length) {
    var c = tape[i];
    if (c == '[') {
      if (!needTeam) return -1;
      depth++;
      i++;
      continue;
    }
    if (c == ']') {
      if (needTeam || sawArrow || depth == 0) return -1;
      depth--;
      i++;
      continue;
    }
    if (c == ',') {
      if (needTeam || sawArrow) return -1;
      needTeam = true;
      left = '';
      i++;
      continue;
    }
    if (c == '>') {
      if (left.isEmpty || sawArrow) return -1;
      sawArrow = true;
      needTeam = true;
      i++;
      continue;
    }
    int code = c.codeUnitAt(0);
    if (code >= 65 && code <= 90) {
      int start = i;
      while (i < tape.length) {
        int next = tape.codeUnitAt(i);
        if (next < 65 || next > 90) break;
        i++;
      }
      String team = tape.substring(start, i);
      if (!needTeam) return -1;
      if (sawArrow) {
        score += left.compareTo(team) > 0 ? depth * 2 : depth;
        sawArrow = false;
      } else {
        left = team;
      }
      needTeam = false;
      continue;
    }
    return -1;
  }
  return (depth == 0 && !needTeam && !sawArrow) ? score : -1;
}

@pragma('vm:entry-point')
void main() {
  assert(scoreBracketTape("") == 0);
  assert(scoreBracketTape("[C>A,B>C]") == 3);
  assert(scoreBracketTape("[A>B,]") == -1);
  print('All tests passed!');
}