@pragma('vm:entry-point')
int serverLogDigitResidue(String logLine) {
  int value = 0;
  for (int i = 0; i < logLine.length; i++) {
    int c = logLine.codeUnitAt(i);
    if (c >= 48 && c <= 57) {
      value = (value * 11 + c - 48) % 97;
    }
  }
  return value;
}

@pragma('vm:entry-point')
void main() {
  assert(serverLogDigitResidue('404') == 3);
  assert(serverLogDigitResidue('user99') == 11);
  assert(serverLogDigitResidue('89') == 0);
  print('All tests passed!');
}