@pragma('vm:entry-point')
int trafficPhaseSyncScore(int phaseCode) {
  int n = phaseCode.abs();
  if (n == 0) return 0;
  List<int> phases = [];
  while (n > 0) {
    int d = n % 10;
    if (d != 0) {
      phases.add(d);
    }
    n ~/= 10;
  }
  if (phases.length < 2) {
    return phases.isEmpty ? 0 : phases[0];
  }
  int score = 0;
  for (int i = 0; i < phases.length; i++) {
    for (int j = i + 1; j < phases.length; j++) {
      int a = phases[i];
      int b = phases[j];
      int x = a;
      int y = b;
      while (y != 0) {
        int t = x % y;
        x = y;
        y = t;
      }
      if (x == 1) {
        score += ((a + b) % 2 == 0) ? 3 : 2;
        continue;
      }
      if (a == b) {
        score += a;
      } else if (((a ~/ x) + (b ~/ x)) % 2 == 0) {
        score -= 1;
      } else {
        score += 1;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(trafficPhaseSyncScore(11) == 3);
  assert(trafficPhaseSyncScore(26) == -1);
  assert(trafficPhaseSyncScore(-272) == 6);
  print('All tests passed!');
}