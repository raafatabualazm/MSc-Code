@pragma('vm:entry-point')
bool validateRecipeScalingLedger(List<String> steps) {
  final List<int> targets = [];
  final List<int> totals = [];
  final List<List<int>> logs = [];
  for (final step in steps) {
    final parts = step.split(' ');
    if (parts.isEmpty || parts[0].isEmpty) return false;
    if (parts[0] == 'batch') {
      if (parts.length != 2) return false;
      final target = int.tryParse(parts[1]);
      if (target == null || target <= 0) return false;
      targets.add(target);
      totals.add(0);
      logs.add([]);
    } else if (parts[0] == 'add') {
      if (parts.length != 2 || targets.isEmpty) return false;
      final amount = int.tryParse(parts[1]);
      if (amount == null || amount <= 0) return false;
      totals[totals.length - 1] += amount;
      logs[logs.length - 1].add(amount);
    } else if (parts[0] == 'undo') {
      if (parts.length != 2 || targets.isEmpty) return false;
      final count = int.tryParse(parts[1]);
      if (count == null || count <= 0 || logs.last.length < count) return false;
      for (int i = 0; i < count; i++) {
        totals[totals.length - 1] -= logs.last.removeLast();
      }
    } else if (parts[0] == 'seal') {
      if (parts.length != 1 || targets.isEmpty) return false;
      final completed = totals.removeLast();
      final expected = targets.removeLast();
      final history = logs.removeLast();
      int check = 0;
      for (final value in history) {
        check += value;
      }
      if (completed != expected || check != expected) return false;
      if (totals.isNotEmpty) {
        totals[totals.length - 1] += completed;
        logs[logs.length - 1].add(completed);
      }
    } else {
      return false;
    }
  }
  return targets.isEmpty;
}

@pragma('vm:entry-point')
void main() {
  assert(validateRecipeScalingLedger([]) == true);
  assert(validateRecipeScalingLedger(['batch 4', 'add 2', 'add 2', 'seal']) == true);
  assert(validateRecipeScalingLedger(['batch 3', 'add 2', 'seal']) == false);
  print('All tests passed!');
}