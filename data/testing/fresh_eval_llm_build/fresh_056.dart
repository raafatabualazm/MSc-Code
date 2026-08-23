@pragma('vm:entry-point')
int alternatingTideAlerts(String log, int limit) {
  int alerts = 0, value = 0, sign = 0, lastSign = 0;
  bool reading = false, hasDigit = false;
  for (int i = 0; i <= log.length; i++) {
    String ch = i < log.length ? log[i] : ';';
    if (!reading && (ch == '+' || ch == '-')) {
      sign = ch == '+' ? 1 : -1;
      value = 0;
      reading = true;
      hasDigit = false;
    } else if (reading && ch.codeUnitAt(0) >= 48 && ch.codeUnitAt(0) <= 57) {
      value = value * 10 + ch.codeUnitAt(0) - 48;
      hasDigit = true;
    } else if (ch == ';') {
      if (reading && hasDigit && value >= limit) {
        if (lastSign != sign) alerts++;
        lastSign = sign;
      } else if (reading) {
        lastSign = 0;
      }
      reading = false;
    } else {
      reading = false;
      lastSign = 0;
    }
  }
  return alerts;
}

@pragma('vm:entry-point')
void main() {
  assert(alternatingTideAlerts("+5;-7;+8;", 5) == 3);
  assert(alternatingTideAlerts("+5;+7;-8;", 5) == 2);
  assert(alternatingTideAlerts("++5;-5;", 5) == 1);
  print('All tests passed!');
}