@pragma('vm:entry-point')
bool inventoryCoversAllBundleSizes(List<int> gemStacks) {
  int covered = 0;
  for (final stack in gemStacks) {
    if (stack > covered + 1) {
      return false;
    }
    covered += stack;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(inventoryCoversAllBundleSizes([]) == true);
  assert(inventoryCoversAllBundleSizes([1, 2, 4]) == true);
  assert(inventoryCoversAllBundleSizes([1, 1, 4]) == false);
  print('All tests passed!');
}