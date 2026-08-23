@pragma('vm:entry-point')
List<List<int>> applyThermostatUndoLog(List<int> commands) {
  final List<List<int>> stack = [];
  for (final cmd in commands) {
    if (cmd > 0) {
      final int hour = (cmd >> 8) & 0xFF;
      final int temp = cmd & 0xFF;
      stack.add([hour, temp]);
    } else if (cmd < 0) {
      final int undoCount = -cmd;
      final int removeCount = undoCount < stack.length ? undoCount : stack.length;
      for (int i = 0; i < removeCount; i++) {
        stack.removeLast();
      }
    }
  }
  return stack;
}

@pragma('vm:entry-point')
void main() {
  assert(applyThermostatUndoLog([]).toString() == '[]');
  assert(applyThermostatUndoLog([0x0A46, 0x0F52, -1]).toString() == '[[10, 70]]');
  assert(applyThermostatUndoLog([0x0A46, 0x0F52, 0x0832, -2, 0x1240]).toString() == '[[10, 70], [18, 64]]');
  print('All tests passed!');
}