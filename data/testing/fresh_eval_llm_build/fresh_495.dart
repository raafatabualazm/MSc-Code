@pragma('vm:entry-point')
String classifyMoisturePathSafety(List<List<int>> grid) {
  if (grid.isEmpty || grid[0].isEmpty) return "No path";
  int rows = grid.length, cols = grid[0].length;
  const int INF = 1000000;
  var dp = List.generate(rows, (_) => List.filled(cols, INF));
  if (grid[0][0] > 50) return "No path";
  dp[0][0] = grid[0][0];
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (i == 0 && j == 0) continue;
      if (grid[i][j] > 50) continue;
      int up = (i > 0) ? dp[i-1][j] : INF;
      int left = (j > 0) ? dp[i][j-1] : INF;
      int best = up < left ? up : left;
      if (best != INF) dp[i][j] = best + grid[i][j];
    }
  }
  int minTotal = dp[rows-1][cols-1];
  if (minTotal == INF) return "No path";
  if (minTotal <= 100) return "Safe path";
  if (minTotal <= 200) return "Caution: moist";
  return "Danger: saturated";
}

@pragma('vm:entry-point')
void main() {
  assert(classifyMoisturePathSafety([]) == "No path");
  assert(classifyMoisturePathSafety([[50]]) == "Safe path");
  assert(classifyMoisturePathSafety([[51]]) == "No path");
  print('All tests passed!');
}