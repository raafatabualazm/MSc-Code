@pragma('vm:entry-point')
num satelliteSwathCoverageSum(List<List<int>> grid) {
  if (grid.isEmpty) return 0;
  int target = grid.length - 1;
  int total = 0;
  for (int r = 0; r < grid.length; r++) {
    int c = target - r;
    if (c >= 0 && c < grid[r].length) total += grid[r][c];
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(satelliteSwathCoverageSum([]) == 0);
  assert(satelliteSwathCoverageSum([[1,2],[3,4]]) == 5);
  assert(satelliteSwathCoverageSum([[1,2,3],[4,5,6],[7,8,9]]) == 15);
  print('All tests passed!');
}