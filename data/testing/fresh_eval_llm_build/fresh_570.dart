@pragma('vm:entry-point')
int countRiskyChargeCycles(String ledger, int threshold) {
  int riskyCount = 0;
  for (String segment in ledger.split('|')) {
    if (segment.trim().isEmpty) continue;
    int level = 0, repeat = 0;
    bool risky = false, invalid = false;
    String prev = '';
    for (String token in segment.split(' ')) {
      if (token.isEmpty) continue;
      String op = token[0];
      if (op == 'H') {
        if (token.length != 1) { invalid = true; break; }
        repeat = 0;
        continue;
      }
      int value = 0;
      for (int i = 1; i < token.length; i++) {
        int code = token.codeUnitAt(i) - 48;
        if (code < 0 || code > 9 || (i == 1 && code == 0 && token.length > 2)) { invalid = true; break; }
        value = value * 10 + code;
      }
      if (invalid || value == 0 || (op != 'C' && op != 'D')) break;
      level += op == 'C' ? value : -value;
      repeat = (op == prev && value > threshold ~/ 2) ? repeat + 1 : 0;
      prev = op;
      if (level.abs() > threshold || repeat >= 2) risky = true;
      if (risky && level == 0) break;
    }
    if (!invalid && risky) riskyCount++;
  }
  return riskyCount;
}

@pragma('vm:entry-point')
void main() {
  assert(countRiskyChargeCycles('', 5) == 0);
  assert(countRiskyChargeCycles('C5', 5) == 0);
  assert(countRiskyChargeCycles('C6 D6', 5) == 1);
  print('All tests passed!');
}