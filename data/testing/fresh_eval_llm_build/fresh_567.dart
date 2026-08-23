@pragma('vm:entry-point')
int pantryFermentationCarry(List<int> plan) {
  int score = 0;
  int lastDay = -1000;
  for (int i = 0; i + 2 < plan.length; i += 3) {
    int start = plan[i];
    int days = plan[i + 1];
    int scale = plan[i + 2];
    if (days < 0) {
      score -= scale.abs();
      continue;
    }
    if (days == 0) {
      if (scale > 0 && start % 2 == 0) {
        score += scale;
      }
      continue;
    }
    for (int d = 0; d < days; d++) {
      int day = start + d;
      int portion = scale;
      if (day - lastDay > 3) {
        score += 1;
      } else if (day == lastDay) {
        portion ~/= 2;
      }
      if (day % 5 == 0) {
        portion += 2;
      } else if ((day + scale) % 3 == 0) {
        portion -= 1;
      }
      if (portion <= 0) {
        continue;
      }
      score += portion;
      lastDay = day;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(pantryFermentationCarry([]) == 0);
  assert(pantryFermentationCarry([2, 0, 4]) == 4);
  assert(pantryFermentationCarry([5, 1, 2, 5, 1, 2]) == 8);
  print('All tests passed!');
}