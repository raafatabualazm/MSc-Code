@pragma('vm:entry-point')
int bracketRecoveryPenalty(List<int> roundDays, int resetWindow) {
  if (resetWindow <= 0) {
    return -1;
  }
  if (roundDays.isEmpty) {
    return 0;
  }
  int score = 0;
  for (int i = 0; i < roundDays.length; i++) {
    int day = roundDays[i];
    if (day < 0) {
      score += 2;
      continue;
    }
    int nearestGap = 1 << 30;
    for (int j = i + 1; j < roundDays.length; j++) {
      int gap = roundDays[j] - day;
      if (gap < 0) {
        score += 4;
        continue;
      }
      if (gap < nearestGap) {
        nearestGap = gap;
      }
      if (gap == 0) {
        score += 7;
      } else if (gap < resetWindow) {
        score += resetWindow - gap;
      } else if (gap % resetWindow == 0) {
        score -= 1;
      }
      if (gap > resetWindow * 2) {
        break;
      }
    }
    if (nearestGap == (1 << 30)) {
      score += day % 3;
    } else if (nearestGap == 1) {
      score += 5;
    } else if (nearestGap > resetWindow) {
      score += 2;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(bracketRecoveryPenalty([], 3) == 0);
  assert(bracketRecoveryPenalty([1, 4], 3) == 0);
  assert(bracketRecoveryPenalty([5, 2], 3) == 8);
  print('All tests passed!');
}