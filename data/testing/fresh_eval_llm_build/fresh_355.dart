@pragma('vm:entry-point')
String evaluateDiceRoundTimeline(List<int> gaps) {
  int day = 0;
  int streak = 0;
  int bestStreak = 0;
  int monthWraps = 0;
  for (final gap in gaps) {
    if (gap <= 0) {
      streak = 0;
    } else {
      int previous = day;
      day += gap;
      if (gap == 1) {
        streak++;
      } else if (gap <= 3) {
        streak = 1;
      } else {
        streak = 0;
      }
      if (streak > bestStreak) bestStreak = streak;
      if (previous ~/ 30 != day ~/ 30) monthWraps++;
    }
  }
  int idleDays = 0;
  for (final gap in gaps) {
    if (gap > 3) idleDays += gap - 3;
  }
  return '$day|$bestStreak|$monthWraps|$idleDays';
}

@pragma('vm:entry-point')
void main() {
  assert(evaluateDiceRoundTimeline([]) == '0|0|0|0');
  assert(evaluateDiceRoundTimeline([1, 1, 1]) == '3|3|0|0');
  assert(evaluateDiceRoundTimeline([30]) == '30|0|1|27');
  print('All tests passed!');
}