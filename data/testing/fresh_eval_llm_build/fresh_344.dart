@pragma('vm:entry-point')
int layeredMoistureRetention(List<List<int>> grid) {
  if (grid.isEmpty) return 0;
  int cols = grid[0].length;
  for (final row in grid) {
    if (row.length != cols) return -1;
  }
  int total = 0;
  List<List<int>> dp =
      List.generate(grid.length, (_) => List.filled(cols, 0));
  for (int i = 0; i < grid.length; i++) {
    for (int j = 0; j < cols; j++) {
      int v = grid[i][j];
      if (v <= 0) continue;
      int best = 1;
      if (i > 0 && grid[i - 1][j] <= v && v - grid[i - 1][j] <= 2) {
        best = dp[i - 1][j] + 1;
      }
      if (j > 0 && grid[i][j - 1] <= v && v - grid[i][j - 1] <= 2) {
        int cand = dp[i][j - 1] + 1;
        best = cand > best ? cand : best;
      }
      if (i > 0 &&
          j > 0 &&
          dp[i - 1][j - 1] > 1 &&
          (grid[i - 1][j - 1] - v).abs() <= 1) {
        best++;
      }
      dp[i][j] = best;
      if (best >= 3) {
        total += best;
      } else if (best == 2 && (i + j).isEven) {
        total += 1;
      }
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(layeredMoistureRetention([]) == 0);
  assert(layeredMoistureRetention([[1, 2, 3]]) == 3);
  assert(layeredMoistureRetention([[1, 2, 2], [1, 2, 3]]) == 11);
  print('All tests passed!');
}