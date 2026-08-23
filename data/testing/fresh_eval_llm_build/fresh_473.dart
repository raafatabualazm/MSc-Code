@pragma('vm:entry-point')
int mazeCellGateScore(int cells) {
  if (cells == 0) {
    return 0;
  }
  int n = cells.abs();
  int score = 0;
  for (int rows = 1; rows * rows <= n; rows++) {
    if (n % rows != 0) {
      continue;
    }
    int cols = n ~/ rows;
    int a = rows;
    int b = cols;
    while (b != 0) {
      int t = a % b;
      a = b;
      b = t;
    }
    int g = a;
    if (g == 1) {
      score += cols - rows;
      continue;
    }
    bool prime = g > 1;
    for (int d = 2; d * d <= g; d++) {
      if (g % d == 0) {
        prime = false;
        score += d % 2 == 0 ? d : -d;
        break;
      }
    }
    if (prime) {
      score += g * (rows == cols ? 2 : 1);
    } else if (g % 2 == 0) {
      score += 1;
    }
  }
  return cells < 0 ? -score : score;
}

@pragma('vm:entry-point')
void main() {
  assert(mazeCellGateScore(0) == 0);
  assert(mazeCellGateScore(4) == 7);
  assert(mazeCellGateScore(-16) == -20);
  print('All tests passed!');
}