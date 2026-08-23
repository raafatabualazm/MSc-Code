@pragma('vm:entry-point')
int batteryCycleIntervalScore(List<int> deltas) {
  if (deltas.isEmpty) return 0;
  int cumulative = 0, lastCycleDay = -1, score = 0;
  for (int i = 0; i < deltas.length; i++) {
    cumulative += deltas[i];
    if (deltas[i] == 0) {
      score -= 1;
      continue;
    }
    while (cumulative >= 100) {
      int gap = lastCycleDay == -1 ? i + 1 : i - lastCycleDay;
      if (gap == 1) score += 5;
      else if (gap <= 3) score += gap;
      else score -= gap;
      for (int j = lastCycleDay + 1; j < i; j++) {
        if (deltas[j] < 0) score -= 1;
        else if (deltas[j] > 50) score += 2;
      }
      lastCycleDay = i;
      cumulative -= 100;
    }
    while (cumulative <= -100) {
      score -= 7;
      cumulative += 100;
      lastCycleDay = i;
    }
  }
  return cumulative.abs() >= 50 ? score + 1 : score - 1;
}

@pragma('vm:entry-point')
void main() {
  assert(batteryCycleIntervalScore([100]) == 4);
  assert(batteryCycleIntervalScore([25, 25, 25, 25]) == -5);
  assert(batteryCycleIntervalScore([-120]) == -8);
  print('All tests passed!');
}