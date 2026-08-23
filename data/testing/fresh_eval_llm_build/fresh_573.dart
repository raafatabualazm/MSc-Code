@pragma('vm:entry-point')
double thermostatCycleBalance(String schedule) {
  if (schedule.isEmpty) return 0.0;
  double total = 0.0;
  for (String block in schedule.split('|')) {
    if (block.isEmpty) continue;
    int value = 0;
    bool valid = true;
    for (int i = 0; i < block.length; i++) {
      int d = block.codeUnitAt(i) - 48;
      if (d < 0 || d > 5) {
        valid = false;
        break;
      }
      value = value * 6 + d;
    }
    if (!valid) continue;
    if (value < 2) {
      total -= 0.5;
      continue;
    }
    int factors = 0, remaining = value;
    for (int p = 2; p * p <= remaining; p++) {
      if (remaining % p != 0) continue;
      int power = 0;
      while (remaining % p == 0) {
        remaining ~/= p;
        power++;
      }
      total += p.isEven ? power.toDouble() : power / 2;
      factors++;
    }
    if (remaining > 1) {
      total += remaining.isEven ? 1.0 : 0.5;
      factors++;
    }
    if (factors > 2 && value % 6 == 0) {
      total += 0.5;
    } else if (factors == 1 && value % 5 == 0) {
      total -= 1.0;
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(thermostatCycleBalance('10') == 1.5);
  assert(thermostatCycleBalance('18|5') == -0.5);
  assert(thermostatCycleBalance('50') == 2.5);
  print('All tests passed!');
}