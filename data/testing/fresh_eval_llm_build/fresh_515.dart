@pragma('vm:entry-point')
String auditDiceRoundUndo(List<String> events, int cap) {
  List<int> stack = [];
  for (String e in events) {
    if (e == 'undo') {
      if (stack.isNotEmpty) stack.removeLast();
      continue;
    }
    if (e == 'double') {
      if (stack.isEmpty) return 'invalid';
      stack.add(stack.last * 2);
      continue;
    }
    if (e.startsWith('drop')) {
      int count = int.parse(e.substring(4));
      while (count > 0 && stack.isNotEmpty) {
        stack.removeLast();
        count--;
      }
      continue;
    }
    if (!e.startsWith('[') || !e.endsWith(']')) return 'invalid';
    int sum = 0;
    for (String part in e.substring(1, e.length - 1).split(',')) {
      int die = int.parse(part);
      if (die < 1 || die > 6) return 'invalid';
      if (sum + die > cap) break;
      sum += die;
    }
    if (sum > 0) stack.add(sum);
  }
  int total = 0;
  int hot = 0;
  for (int v in stack) {
    total += v;
    if (v >= cap) {
      hot++;
    } else if (hot > 0) {
      hot--;
    }
  }
  return '$total|$hot';
}

@pragma('vm:entry-point')
void main() {
  assert(auditDiceRoundUndo([], 5) == '0|0');
  assert(auditDiceRoundUndo(['[2,3]', 'double'], 5) == '15|2');
  assert(auditDiceRoundUndo(['double'], 4) == 'invalid');
  print('All tests passed!');
}