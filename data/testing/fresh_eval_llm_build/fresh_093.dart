@pragma('vm:entry-point')
bool verifyBatteryCycleReplay(List<String> log) {
  int level = 50;
  bool sawDrain = false;
  List<int> stack = [];
  for (String entry in log) {
    if (entry == 'S') {
      stack.add(level);
      continue;
    }
    if (entry == 'R') {
      if (stack.isEmpty || level == stack.last) return false;
      level = stack.removeLast();
      continue;
    }
    if (entry.length < 2) return false;
    int value = 0;
    for (int i = 1; i < entry.length; i++) {
      int code = entry.codeUnitAt(i);
      if (code < 48 || code > 57) return false;
      value = value * 10 + code - 48;
      if (value > 40) return false;
    }
    if (entry[0] == 'C') {
      if (value < 1 || value > 25 || level + value > 100) return false;
      level += value;
    } else if (entry[0] == 'D') {
      sawDrain = true;
      if (value < 1 || value > 20 || level - value < 0) return false;
      level -= value;
      int lowDigits = 0;
      for (int t = level; t > 0; t ~/= 10) {
        if (t % 10 < 3) lowDigits++;
      }
      if (lowDigits > 1 && level < 15) return false;
    } else {
      return false;
    }
  }
  return sawDrain && stack.isEmpty && level >= 10;
}

@pragma('vm:entry-point')
void main() {
  assert(verifyBatteryCycleReplay(['D10']) == true);
  assert(verifyBatteryCycleReplay(['S', 'R']) == false);
  assert(verifyBatteryCycleReplay(['D19', 'D19']) == false);
  print('All tests passed!');
}