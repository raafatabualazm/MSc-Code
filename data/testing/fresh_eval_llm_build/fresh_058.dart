@pragma('vm:entry-point')
List<String> filterRareTierItems(List<String> inventory) {
  if (inventory.isEmpty) return [];
  final List<List<int>> rareTiers = [[16, 32], [64, 128], [256, 512]];
  // Parse powers
  final List<int> powers = [];
  for (final item in inventory) {
    final colon = item.indexOf(':');
    if (colon < 0) { powers.add(-1); continue; }
    powers.add(int.tryParse(item.substring(colon + 1)) ?? -1);
  }
  final List<String> result = [];
  for (final tier in rareTiers) {
    final int lo = tier[0], hi = tier[1];
    // Binary search for first index where power >= lo
    int left = 0, right = powers.length;
    while (left < right) {
      final mid = (left + right) >> 1;
      if (powers[mid] < lo) { left = mid + 1; } else { right = mid; }
    }
    int start = left;
    // Binary search for first index where power >= hi
    left = start; right = powers.length;
    while (left < right) {
      final mid = (left + right) >> 1;
      if (powers[mid] < hi) { left = mid + 1; } else { right = mid; }
    }
    int end = left;
    for (int i = start; i < end; i++) {
      final item = inventory[i];
      final colon = item.indexOf(':');
      if (colon <= 0) continue;
      final name = item.substring(0, colon);
      if (name.startsWith('x') || name.startsWith('z')) continue;
      result.add(name);
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(filterRareTierItems([]).toString() == '[]');
  assert(filterRareTierItems(['sword:20', 'bow:65', 'axe:300']).toString() == '[sword, bow, axe]');
  assert(filterRareTierItems(['xenon:70', 'zarc:100']).toString() == '[]');
  print('All tests passed!');
}