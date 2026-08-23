@pragma('vm:entry-point')
bool validateMorsePulseChecksum(String tape) {
  if (tape.isEmpty) return false;
  int segment = 0, dots = 0, dashes = 0, run = 0;
  String prev = '';
  for (int i = 0; i < tape.length; i++) {
    String ch = tape[i];
    if (ch == '|') {
      if (prev == '|' || dots + dashes == 0) return false;
      if ((segment.isEven && dots <= dashes) || (segment.isOdd && dashes <= dots)) return false;
      segment++; dots = 0; dashes = 0; run = 0; prev = '|';
    } else if (ch == '.' || ch == '-') {
      run = ch == prev ? run + 1 : 1;
      if (run > 3) return false;
      if (ch == '.') { dots++; } else { dashes++; }
      prev = ch;
    } else {
      return false;
    }
  }
  if (prev == '|') return false;
  return !((segment.isEven && dots <= dashes) || (segment.isOdd && dashes <= dots));
}

@pragma('vm:entry-point')
void main() {
  assert(validateMorsePulseChecksum("..|--") == true);
  assert(validateMorsePulseChecksum(". --") == false);
  assert(validateMorsePulseChecksum("....|---") == false);
  print('All tests passed!');
}