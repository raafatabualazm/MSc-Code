@pragma('vm:entry-point')
int satelliteWindowRecoveryIndex(List<int> segments) {
  int score = 0;
  int streak = 0;
  for (final v in segments) {
    if (v > 0) {
      if (v >= 6) {
        score += v ~/ 2;
      } else if (v >= 3) {
        score += 2;
      } else {
        score += 1;
      }
      streak++;
    } else if (v < 0) {
      int gap = -v;
      if (streak > 0) {
        if (gap == 1) {
          score += streak;
        } else if (gap <= 3) {
          score -= 1;
        } else {
          score -= gap;
          streak = 0;
        }
      } else {
        score -= gap ~/ 2;
      }
    }
  }
  return score < 0 ? 0 : score;
}

@pragma('vm:entry-point')
void main() {
  assert(satelliteWindowRecoveryIndex([]) == 0);
  assert(satelliteWindowRecoveryIndex([2, -1, 2]) == 3);
  assert(satelliteWindowRecoveryIndex([6, -4, 6]) == 2);
  print('All tests passed!');
}