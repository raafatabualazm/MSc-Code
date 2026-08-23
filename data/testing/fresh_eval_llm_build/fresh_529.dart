@pragma('vm:entry-point')
int trafficPhaseConflictIndex(List<List<int>> grid) {
  if (grid.isEmpty) return 0;
  int score = 0;
  for (int r = 0; r < grid.length; r++) {
    if (grid[r].isEmpty) continue;
    for (int c = 0; c < grid[r].length; c++) {
      int phase = grid[r][c];
      if (phase < 0 || phase > 2) {
        score += 5;
        continue;
      }
      if ((r == 0 || c == 0) && phase == 1) score += 1;
      if (c > 0) {
        int left = grid[r][c - 1];
        if (left >= 0 && left <= 2) {
          int diff = (phase - left).abs();
          if (diff == 2) score += 3;
          else if (diff == 0) score -= 1;
          else score += 1;
        }
      }
      if (r > 0 && c < grid[r - 1].length) {
        int up = grid[r - 1][c];
        if (up == 2 && phase == 0) score += 4;
        else if (up == 0 && phase == 2) score += 2;
        else if (up == phase) score -= 1;
      }
    }
  }
  return score < 0 ? 0 : score;
}

@pragma('vm:entry-point')
void main() {
  assert(trafficPhaseConflictIndex([]) == 0);
  assert(trafficPhaseConflictIndex([[0, 2]]) == 3);
  assert(trafficPhaseConflictIndex([[0, 0], [0, 0]]) == 0);
  print('All tests passed!');
}