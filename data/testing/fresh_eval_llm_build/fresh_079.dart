@pragma('vm:entry-point')
List<int> inventoryRecoveryWindows(List<int> changes) {
  int stock = 0;
  int streak = 0;
  List<int> result = [];
  for (final change in changes) {
    stock += change;
    streak = stock < 0 ? 0 : streak + 1;
    result.add(streak);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(inventoryRecoveryWindows([]).toString() == '[]');
  assert(inventoryRecoveryWindows([2, -4, 3]).toString() == '[1, 0, 1]');
  assert(inventoryRecoveryWindows([0, 0, -1, 1]).toString() == '[1, 2, 0, 1]');
  print('All tests passed!');
}