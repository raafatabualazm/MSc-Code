@pragma('vm:entry-point')
List<int> scanBatteryChargeCycles(String script) {
  List<int> out = [];
  int level = 0;
  bool draining = false;
  bool fault = false;
  int i = 0;
  while (i < script.length) {
    String ch = script[i];
    if (ch == '#') break;
    if (ch == '|') {
      out.add(fault ? -1 : level);
      level = 0;
      draining = false;
      fault = false;
      i++;
      continue;
    }
    if (ch == 'C' || ch == 'D') {
      int j = i + 1;
      int value = 0;
      while (j < script.length) {
        int d = script.codeUnitAt(j) - 48;
        if (d < 0 || d > 9) break;
        value = value * 10 + d;
        j++;
      }
      if (j == i + 1) {
        fault = true;
        i++;
        continue;
      }
      if (ch == 'C') {
        if (draining && value > 0) fault = true;
        level += value;
      } else {
        draining = true;
        level -= value;
      }
      if (level < 0 || level > 100) fault = true;
      i = j;
    } else if (ch == '!') {
      level = 50;
      draining = false;
      i++;
    } else {
      fault = true;
      i++;
    }
  }
  if (i > 0 && script[script.length - 1] != '|') {
    out.add(fault ? -1 : level);
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(scanBatteryChargeCycles('C10D5|').toString() == '[5]');
  assert(scanBatteryChargeCycles('C5||').toString() == '[5, 0]');
  assert(scanBatteryChargeCycles('').length == 0);
  print('All tests passed!');
}