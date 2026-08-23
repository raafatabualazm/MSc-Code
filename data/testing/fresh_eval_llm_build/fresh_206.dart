@pragma('vm:entry-point')
List<int> simulateWarehouseStack(List<int> commands) {
  var stack = <int>[];
  for (int cmd in commands) {
    if (cmd > 0) {
      stack.add(cmd);
    } else if (cmd == 0) {
      if (stack.isNotEmpty) stack.removeLast();
    } else if (cmd == -1) {
      if (stack.isNotEmpty) stack.add(stack.last);
    } else if (cmd == -2) {
      if (stack.length >= 2) {
        int top = stack.last;
        stack[stack.length - 1] = stack[stack.length - 2];
        stack[stack.length - 2] = top;
      }
    }
  }
  return stack;
}

@pragma('vm:entry-point')
void main() {
  assert(simulateWarehouseStack([]).isEmpty);
  assert(simulateWarehouseStack([5, 3, 8]).length == 3);
  assert(simulateWarehouseStack([1, 2, 0]).first == 1);
  print('All tests passed!');
}