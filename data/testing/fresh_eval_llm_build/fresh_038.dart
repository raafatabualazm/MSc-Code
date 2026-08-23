@pragma('vm:entry-point')
int qrModuleReprintDelayScore(List<int> moduleDays) {
  int score = 0;
  for (int i = 1; i < moduleDays.length; i++) {
    int gap = moduleDays[i] - moduleDays[i - 1];
    if (gap == 0) {
      score += 2;
    } else if (gap > 0) {
      if (gap <= 3) {
        score += gap * 3;
      } else {
        score += 7 + (gap % 4);
      }
    } else {
      if (gap == -1) {
        score -= 2;
      } else {
        score += gap;
      }
    }
  }
  for (int day in moduleDays) {
    if (day % 7 == 0) {
      score += 1;
    } else if (day < 0 && day.isEven) {
      score -= 1;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(qrModuleReprintDelayScore([]) == 0);
  assert(qrModuleReprintDelayScore([2, 6]) == 7);
  assert(qrModuleReprintDelayScore([7, 7, 8]) == 7);
  print('All tests passed!');
}