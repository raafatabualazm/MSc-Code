@pragma('vm:entry-point')
int bracketAdvanceChecksum(String tape) {
  int score = 0;
  String last = '';
  int streak = 0;
  for (int i = 0; i < tape.length; i++) {
    String ch = tape[i];
    int code = ch.codeUnitAt(0);
    if (ch == '!') {
      last = '';
      streak = 0;
    } else if (code >= 50 && code <= 57 && last.isNotEmpty) {
      int extra = code - 48;
      for (int j = 1; j < extra; j++) {
        streak++;
        int seed = last.codeUnitAt(0) - 64;
        score += streak.isEven ? seed : -1;
      }
    } else if (code >= 65 && code <= 90) {
      last = ch;
      streak = 1;
      score += code - 64;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(bracketAdvanceChecksum('A2') == 2);
  assert(bracketAdvanceChecksum('B4') == 5);
  assert(bracketAdvanceChecksum('A!A2') == 3);
  print('All tests passed!');
}