@pragma('vm:entry-point')
import 'dart:math';

bool evaluateBatteryHealth(String events) {
  int charge = 100;
  int deepCycles = 0;
  for (int i = 0; i < events.length; i++) {
    String c = events[i];
    if (c == '+') {
      charge = min(100, charge + 10);
    } else if (c == '-') {
      charge = max(0, charge - 5);
    } else if (c == '*') {
      if (charge <= 20) return false;
      deepCycles++;
      if (deepCycles > 3) return false;
      while (charge > 0) {
        charge = max(0, charge - 5);
      }
      while (charge < 100) {
        charge = min(100, charge + 10);
      }
    } else {
      return false;
    }
  }
  return charge >= 20;
}

@pragma('vm:entry-point')
void main() {
  assert(evaluateBatteryHealth('') == true);
  assert(evaluateBatteryHealth('****') == false);
  assert(evaluateBatteryHealth('***') == true);
  print('All tests passed!');
}