@pragma('vm:entry-point')
int countItemsByRarityLength(String inventory, String separator, int minLength) {
  if (inventory.isEmpty) return 0;
  final tokens = inventory.split(separator);
  int count = 0;
  for (final token in tokens) {
    final parts = token.split(':');
    if (parts.length == 2 && parts[1].length > minLength) {
      count++;
    }
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countItemsByRarityLength('sword:rare,potion:common,shield:rare', ',', 3) == 3);
  assert(countItemsByRarityLength('sword:rare,potion:common,shield:rare', ',', 4) == 1);
  assert(countItemsByRarityLength('', ',', 0) == 0);
  print('All tests passed!');
}