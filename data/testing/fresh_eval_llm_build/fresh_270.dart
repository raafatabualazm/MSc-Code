@pragma('vm:entry-point')
bool isCorrectPhaseEncoding(String s) {
  if (s.isEmpty || s.length % 2 != 0) return false;
  int expected = 0;
  for (int i = 0; i < s.length; i += 2) {
    String letter = s[i];
    String digit = s[i+1];
    if (letter != 'G' && letter != 'Y' && letter != 'R') return false;
    if (digit.codeUnitAt(0) < 49 || digit.codeUnitAt(0) > 57) return false;
    if ('GYR'.indexOf(letter) != expected) return false;
    expected = (expected + 1) % 3;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isCorrectPhaseEncoding('G5') == true);
  assert(isCorrectPhaseEncoding('Y2') == false);
  assert(isCorrectPhaseEncoding('G0') == false);
  print('All tests passed!');
}