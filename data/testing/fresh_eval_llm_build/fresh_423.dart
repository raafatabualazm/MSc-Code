@pragma('vm:entry-point')
int rgbDiamondConflictScore(List<List<int>> pixels) {
  int score = 0;
  for (int i = 0; i < pixels.length; i++) {
    var p = pixels[i];
    int radius = 0;
    for (int c = 2; c < 5; c++) {
      if (p[c] >= 200) {
        radius++;
      } else if (p[c] <= 20) {
        score -= 1;
      }
    }
    score += 1 + 2 * radius * (radius + 1);
    for (int j = 0; j < i; j++) {
      var q = pixels[j];
      int qRadius = ((q[2] >= 200) ? 1 : 0) + ((q[3] >= 200) ? 1 : 0) + ((q[4] >= 200) ? 1 : 0);
      int dist = (p[0] - q[0]).abs() + (p[1] - q[1]).abs();
      if (dist <= radius + qRadius) {
        score += 5;
      } else if (dist == radius + qRadius + 1) {
        score += 1;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(rgbDiamondConflictScore([]) == 0);
  assert(rgbDiamondConflictScore([[0, 0, 255, 255, 255]]) == 25);
  assert(rgbDiamondConflictScore([[0, 0, 255, 0, 0], [3, 0, 255, 0, 0]]) == 7);
  print('All tests passed!');
}