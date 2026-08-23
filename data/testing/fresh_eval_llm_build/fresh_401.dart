@pragma('vm:entry-point')
bool validateTideDriftLog(String log) {
  if (log.isEmpty) return false;
  int i = 0;
  while (i < log.length) {
    String mode = log[i];
    if (mode != 'R' && mode != 'F') return false;
    i++;
    String lastSign = '';
    int groups = 0;
    while (i < log.length && log[i] != '|') {
      if (i + 2 >= log.length) return false;
      String sign = log[i];
      int tens = log.codeUnitAt(i + 1) - 48;
      int half = log.codeUnitAt(i + 2) - 48;
      if ((sign != '+' && sign != '-') || tens < 0 || tens > 3) return false;
      if (half != 0 && half != 5) return false;
      if (sign == lastSign || (sign == '-' && tens == 0 && half == 0)) return false;
      lastSign = sign;
      groups++;
      i += 3;
      if (i < log.length && log[i] != '|' && log[i] != '+' && log[i] != '-') {
        return false;
      }
    }
    if (groups == 0) return false;
    if ((mode == 'R' && lastSign != '+') || (mode == 'F' && lastSign != '-')) {
      return false;
    }
    if (i < log.length) i++;
    if (i == log.length && log.endsWith('|')) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(validateTideDriftLog('R+10') == true);
  assert(validateTideDriftLog('R+10-05') == false);
  assert(validateTideDriftLog('R+10|F-35') == true);
  print('All tests passed!');
}