@pragma('vm:entry-point')
String reconcileRoundedCentTape(String tape) {
  List<int> stack = [];
  List<List<int>> history = [];
  List<String> tokens = tape.trim().isEmpty ? [] : tape.split(RegExp(r'\s+'));
  for (String token in tokens) {
    if (token == '!') {
      if (history.isNotEmpty) stack = List<int>.from(history.removeLast());
    } else if (token == 'R') {
      if (stack.isNotEmpty) {
        history.add(List<int>.from(stack));
        int v = stack.removeLast();
        int rem = v.abs() % 5;
        if (rem >= 3) {
          v += v >= 0 ? 5 - rem : -(5 - rem);
        } else if (rem != 0) {
          v += v >= 0 ? -rem : rem;
        }
        stack.add(v);
      }
    } else if (token == '+') {
      if (stack.length > 1) {
        history.add(List<int>.from(stack));
        stack.add(stack.removeLast() + stack.removeLast());
      }
    } else {
      history.add(List<int>.from(stack));
      stack.add(int.parse(token));
    }
  }
  return stack.isEmpty ? 'empty' : stack.join('|');
}

@pragma('vm:entry-point')
void main() {
  assert(reconcileRoundedCentTape('12 R') == '10');
  assert(reconcileRoundedCentTape('8 9 +') == '17');
  assert(reconcileRoundedCentTape('5 !') == 'empty');
  print('All tests passed!');
}