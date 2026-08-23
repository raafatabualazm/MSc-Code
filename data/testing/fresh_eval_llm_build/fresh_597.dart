@pragma('vm:entry-point')
int wifiBinInterferenceScore(List<List<int>> grid) {
  if (grid.isEmpty) return 0;
  int score = 0;
  int binOf(int v) {
    if (v <= 1) return 0;
    if (v <= 4) return 1;
    if (v <= 7) return 2;
    return 3;
  }

  for (int r = 0; r < grid.length; r++) {
    if (grid[r].isEmpty) continue;
    for (int c = 0; c < grid[r].length; c++) {
      int v = grid[r][c];
      if (v < 0) continue;
      int bin = binOf(v), mismatch = 0;
      bool severe = false;
      for (List<int> d in const [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        int nr = r + d[0], nc = c + d[1];
        if (nr < 0 || nr >= grid.length || nc < 0 || nc >= grid[nr].length) continue;
        int nv = grid[nr][nc];
        if (nv < 0) continue;
        int nbin = binOf(nv);
        if (nbin != bin) mismatch++;
        if ((nbin - bin).abs() >= 2) severe = true;
      }
      if (mismatch == 0) {
        if (bin == 2) score += 2;
        continue;
      }
      if (severe && mismatch >= 2) score += 3;
      else if (mismatch >= 2) score += 1;
      else if (bin == 0 && mismatch == 1) score--;
    }
  }
  return score < 0 ? 0 : score;
}

@pragma('vm:entry-point')
void main() {
  assert(wifiBinInterferenceScore([]) == 0);
  assert(wifiBinInterferenceScore([[5]]) == 2);
  assert(wifiBinInterferenceScore([[2, 8], [8, 2]]) == 12);
  print('All tests passed!');
}