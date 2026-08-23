@pragma('vm:entry-point')
int inventoryOverflowCredits(List<List<int>> itemGrid, int slotLimit) {
  int credits = 0;
  for (final row in itemGrid) {
    final used = row.fold(0, (a, b) => a + b.abs());
    if (used > slotLimit) {
      credits += used - slotLimit;
    }
  }
  return credits;
}

@pragma('vm:entry-point')
void main() {
  assert(inventoryOverflowCredits([], 5) == 0);
  assert(inventoryOverflowCredits([[1, 2], [3]], 2) == 2);
  assert(inventoryOverflowCredits([[-2, 1], [4, -1]], 2) == 4);
  print('All tests passed!');
}