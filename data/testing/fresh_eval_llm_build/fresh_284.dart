@pragma('vm:entry-point')
bool closesLedgerTape(String tape) {
  int balance = 0;
  for (int i = 0; i < tape.length; i += 3) {
    if (i + 2 >= tape.length || (tape[i] != '+' && tape[i] != '-') || tape.codeUnitAt(i + 1) < 48 || tape.codeUnitAt(i + 1) > 57 || tape.codeUnitAt(i + 2) < 48 || tape.codeUnitAt(i + 2) > 57) return false;
    int amount = (tape.codeUnitAt(i + 1) - 48) * 10 + tape.codeUnitAt(i + 2) - 48;
    balance += tape[i] == '+' ? amount : -amount;
    if (balance < 0) return false;
  }
  return balance == 0;
}

@pragma('vm:entry-point')
void main() {
  assert(closesLedgerTape('') == true);
  assert(closesLedgerTape('+05-05') == true);
  assert(closesLedgerTape('+05-04') == false);
  print('All tests passed!');
}