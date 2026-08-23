@pragma('vm:entry-point')
List<int> scanRoundedCentTape(String tape) {
  List<int> out = [];
  int i = 0;
  while (i < tape.length) {
    while (i < tape.length && (tape[i] == ' ' || tape[i] == ',')) {
      i++;
    }
    if (i >= tape.length) return out;
    String mode = tape[i];
    if (mode != '=' && mode != '^' && mode != '_') {
      while (i < tape.length && tape[i] != ',') i++;
      continue;
    }
    i++;
    int sign = 1, dollars = 0, cents = 0, mills = 0, seen = 0;
    if (i < tape.length && (tape[i] == '-' || tape[i] == '+')) {
      sign = tape[i] == '-' ? -1 : 1;
      i++;
    }
    while (i < tape.length && tape.codeUnitAt(i) >= 48 && tape.codeUnitAt(i) <= 57) {
      dollars = dollars * 10 + tape.codeUnitAt(i) - 48;
      i++;
    }
    if (i >= tape.length || tape[i] != '.') continue;
    i++;
    while (i < tape.length && seen < 3 && tape.codeUnitAt(i) >= 48 && tape.codeUnitAt(i) <= 57) {
      int d = tape.codeUnitAt(i) - 48;
      if (seen == 0) cents += d * 10; else if (seen == 1) cents += d; else mills = d;
      seen++;
      i++;
    }
    if (seen < 2) continue;
    int value = sign * (dollars * 100 + cents);
    if (mode == '=' && mills >= 5) value += sign;
    if (mode == '^' && mills > 0) value += sign;
    out.add(value);
    while (i < tape.length && tape[i] != ',') i++;
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(scanRoundedCentTape('').toString() == '[]');
  assert(scanRoundedCentTape('=1.235,^-0.001').toString() == '[124, -1]');
  assert(scanRoundedCentTape('_+2.999, bad,=0.005').length == 2);
  print('All tests passed!');
}