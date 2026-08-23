@pragma('vm:entry-point')
bool validateChargeCycleTape(String tape) {
  int level = 50;
  int i = 0;
  bool sawToken = false;
  while (i < tape.length) {
    int sessionMoves = 0;
    if (tape[i] == '#') return false;
    while (i < tape.length && tape[i] != '#') {
      String op = tape[i++];
      if (op != 'C' && op != 'D') return false;
      if (i >= tape.length || tape.codeUnitAt(i) < 48 || tape.codeUnitAt(i) > 57) return false;
      int value = 0;
      while (i < tape.length) {
        int c = tape.codeUnitAt(i);
        if (c < 48 || c > 57) break;
        value = value * 10 + c - 48;
        if (value > 60) return false;
        i++;
      }
      if (value == 0) return false;
      level += op == 'C' ? value : -value;
      sessionMoves++;
      sawToken = true;
      if (level < 0 || level > 100 || (sessionMoves > 1 && level == 50 && op == 'D')) return false;
    }
    if (sessionMoves == 0 || level % 5 != 0) return false;
    if (i < tape.length && (level < 20 || level > 80 || ++i == tape.length)) return false;
  }
  return sawToken && level >= 25;
}

@pragma('vm:entry-point')
void main() {
  assert(validateChargeCycleTape("C10") == true);
  assert(validateChargeCycleTape("D51") == false);
  assert(validateChargeCycleTape("C30#D10") == true);
  print('All tests passed!');
}