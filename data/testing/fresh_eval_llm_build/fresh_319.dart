@pragma('vm:entry-point')
int theaterBufferComplianceScore(List<String> chart) {
  if (chart.isEmpty) return 0;
  int score = 0;
  for (int r = 0; r < chart.length; r++) {
    if (chart[r].isEmpty) continue;
    for (int c = 0; c < chart[r].length; c++) {
      if (chart[r][c] != 'R') continue;
      int emptySides = 0;
      bool crowded = false;
      const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
      for (final d in dirs) {
        int nr = r + d[0], nc = c + d[1];
        if (nr < 0 || nr >= chart.length || nc < 0 || nc >= chart[nr].length) {
          emptySides++;
        } else if (chart[nr][nc] == 'R') {
          crowded = true;
        } else if (chart[nr][nc] == 'E') {
          emptySides++;
        }
      }
      if (crowded) {
        score -= 3;
        continue;
      }
      if (emptySides >= 3) {
        score += 5;
      } else if (emptySides == 2) {
        score += 2;
      } else {
        score += 1;
      }
      if (c == 0 || c == chart[r].length - 1) {
        score++;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(theaterBufferComplianceScore([]) == 0);
  assert(theaterBufferComplianceScore(['R']) == 6);
  assert(theaterBufferComplianceScore(['RR']) == -6);
  print('All tests passed!');
}