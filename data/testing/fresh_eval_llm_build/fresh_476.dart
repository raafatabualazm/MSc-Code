@pragma('vm:entry-point')
List<int> summarizeSatellitePassWindows(List<List<int>> grid) {
  List<int> result = [];
  for (int r = 0; r < grid.length; r++) {
    int score = 0;
    for (int c = 0; c < grid[r].length; c++) {
      int v = grid[r][c];
      if (v <= 0) {
        bool linked = (r > 0 && c < grid[r - 1].length && grid[r - 1][c] > 0) ||
            (c > 0 && grid[r][c - 1] > 0);
        score += linked ? 1 : -1;
      } else if (c > 0 && grid[r][c - 1] == v) {
        score += 2;
      } else if (r > 0 && c < grid[r - 1].length && grid[r - 1][c] < v) {
        score += 3;
      } else {
        score += 1;
      }
    }
    result.add(score);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(summarizeSatellitePassWindows([[1, 1]]).toString() == '[3]');
  assert(summarizeSatellitePassWindows([[0, 1]]).toString() == '[0]');
  assert(summarizeSatellitePassWindows([[1], [2]]).toString() == '[1, 3]');
  print('All tests passed!');
}