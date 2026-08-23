@pragma('vm:entry-point')
bool hasBalancedPassCoverage(List<List<int>> grid, int threshold) {
  if (grid.isEmpty) return false;
  int total = 0;
  for (final row in grid) {
    int rowSum = row.fold(0, (a, b) => a + b);
    if (rowSum == 0) return false;
    total += rowSum;
  }
  return total < threshold;
}

@pragma('vm:entry-point')
void main() {
  assert(hasBalancedPassCoverage([[1,0],[0,1]], 5) == true);
  assert(hasBalancedPassCoverage([[1,1],[0,1]], 3) == false);
  assert(hasBalancedPassCoverage([], 5) == false);
  print('All tests passed!');
}