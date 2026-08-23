@pragma('vm:entry-point')
List<int> rowPeakMazeSignals(List<String> maze, int fade) {
  if (maze.isEmpty) return [];
  int cols = maze[0].length;
  List<List<int>> dp = List.generate(
      maze.length, (_) => List.filled(cols, 0));
  List<int> peaks = [];
  for (int r = 0; r < maze.length; r++) {
    int rowBest = 0;
    for (int c = 0; c < cols; c++) {
      String cell = maze[r][c];
      if (cell == '#') continue;
      int base = cell == '+' ? 3 : 1;
      int best = base;
      if (r > 0 && dp[r - 1][c] > 0) {
        int up = dp[r - 1][c] + base - fade;
        if (maze[r - 1][c] == '+') up++;
        if (up > best) best = up;
      }
      if (c > 0 && dp[r][c - 1] > 0) {
        int left = dp[r][c - 1] + base - (c.isEven ? fade : 1);
        if (maze[r][c - 1] == '+') left++;
        if (left > best) best = left;
      }
      if (best <= 0) continue;
      dp[r][c] = best;
      if (best > rowBest) rowBest = best;
    }
    peaks.add(rowBest);
  }
  return peaks;
}

@pragma('vm:entry-point')
void main() {
  assert(rowPeakMazeSignals([], 2).toString() == '[]');
  assert(rowPeakMazeSignals(['++'], 1).toString() == '[6]');
  assert(rowPeakMazeSignals(['+.#', '.++'], 2).toString() == '[4, 7]');
  print('All tests passed!');
}