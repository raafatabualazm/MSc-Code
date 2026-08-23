@pragma('vm:entry-point')
int tideReadingLedgerScore(List<String> events) {
  List<int> stack = [];
  int penalty = 0;
  for (String e in events) {
    if (e == 'rollback') {
      if (stack.isNotEmpty) {
        int v = stack.removeLast();
        penalty += v < 0 ? -v : v;
      } else {
        penalty += 2;
      }
    } else if (e == 'surge') {
      if (stack.length >= 2) {
        int a = stack.removeLast();
        int b = stack.removeLast();
        stack.add(a - b);
      } else if (stack.isNotEmpty) {
        stack.add(stack.last);
      }
    } else {
      int v = int.parse(e);
      if (stack.isNotEmpty && ((stack.last < 0) != (v < 0))) {
        penalty += 1;
      }
      stack.add(v);
    }
  }
  for (int v in stack) {
    penalty += v.abs().isEven ? 1 : 2;
  }
  return penalty;
}

@pragma('vm:entry-point')
void main() {
  assert(tideReadingLedgerScore([]) == 0);
  assert(tideReadingLedgerScore(['5', 'rollback']) == 5);
  assert(tideReadingLedgerScore(['2', '-2', 'surge']) == 2);
  print('All tests passed!');
}