@pragma('vm:entry-point')
bool hasBalancedBracketRun(String tape) {
  int total = 0;
  for (int i = 0; i < tape.length; i += 2) {
    if (i + 1 >= tape.length || tape.codeUnitAt(i) < 65 || tape.codeUnitAt(i) > 90) return false;
    int n = tape.codeUnitAt(i + 1) - 48;
    if (n < 1 || n > 8 || (i > 0 && tape[i] == tape[i - 2])) return false;
    total += n;
  }
  return total == 16;
}

@pragma('vm:entry-point')
void main() {
  assert(hasBalancedBracketRun('A8B8') == true);
  assert(hasBalancedBracketRun('A7B8') == false);
  assert(hasBalancedBracketRun('A4A4B8') == false);
  print('All tests passed!');
}