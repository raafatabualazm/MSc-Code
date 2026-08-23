@pragma('vm:entry-point')
int keypadPasswordGeometryScore(String password) {
  const xs = [1, 0, 1, 2, 0, 1, 2, 0, 1, 2];
  const ys = [3, 0, 0, 0, 1, 1, 1, 2, 2, 2];
  if (password.isEmpty) return 0;
  int score = 0;
  for (int i = 0; i < password.length; i++) {
    int d = password.codeUnitAt(i) - 48;
    if (d < 0 || d > 9) {
      score -= 2;
      continue;
    }
    for (int j = 0; j < i; j++) {
      int p = password.codeUnitAt(j) - 48;
      if (p < 0 || p > 9) continue;
      int dx = (xs[d] - xs[p]).abs();
      int dy = (ys[d] - ys[p]).abs();
      int dist = dx + dy;
      if (dist == 0) {
        score += 5;
        if (i - j == 1) return score;
      } else if (dist == 1) {
        score += 2;
      } else if (dx == dy) {
        score -= 1;
      }
      for (int k = 0; k < j; k++) {
        int q = password.codeUnitAt(k) - 48;
        if (q < 0 || q > 9) continue;
        if (xs[q] == xs[d] && ys[q] == ys[p] && dx > 0 && dy > 0) {
          score += dx * dy;
          break;
        }
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(keypadPasswordGeometryScore('12') == 2);
  assert(keypadPasswordGeometryScore('111') == 5);
  assert(keypadPasswordGeometryScore('1397') == 2);
  print('All tests passed!');
}