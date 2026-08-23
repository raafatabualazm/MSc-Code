@pragma('vm:entry-point')
double evaluateInventoryUndo(List<String> commands) {
  double total = 0.0;
  List<double> stack = [];
  for (var cmd in commands) {
    if (cmd.startsWith('add ')) {
      stack.add(total);
      total += double.parse(cmd.substring(4));
    } else if (cmd == 'undo' && stack.isNotEmpty) {
      total = stack.removeLast();
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(candidate([]) == 0.0);
  assert(candidate(['add 5', 'undo']) == 0.0);
  assert(candidate(['add 3.5']) == 3.5);
  print('All tests passed!');
}