@pragma('vm:entry-point')
int evaluateWifiSignalRPN(List<String> tokens) {
  List<int> stack = [];
  for (var t in tokens) {
    if (t == 'A') {
      int b = stack.removeLast();
      int a = stack.removeLast();
      stack.add((a + b) ~/ 2);
    } else if (t == 'X') {
      int b = stack.removeLast();
      int a = stack.removeLast();
      stack.add(a > b ? a : b);
    } else if (t == 'N') {
      int b = stack.removeLast();
      int a = stack.removeLast();
      stack.add(a < b ? a : b);
    } else {
      stack.add(int.parse(t));
    }
  }
  return stack.isEmpty ? 0 : stack.last;
}

@pragma('vm:entry-point')
void main() {
  assert(evaluateWifiSignalRPN(["10"]) == 10);
  assert(evaluateWifiSignalRPN(["10", "20", "A"]) == 15);
  assert(evaluateWifiSignalRPN(["-50", "-40", "A"]) == -45);
  print('All tests passed!');
}