@pragma('vm:entry-point')
int countDiceGameRounds(int startDay, int breakDays, int totalDays) {
  if (startDay >= totalDays) return 0;
  int count = 0;
  int current = startDay;
  int cycle = 1 + breakDays;
  while (current < totalDays) {
    count++;
    current += cycle;
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countDiceGameRounds(0, 1, 5) == 3);
  assert(countDiceGameRounds(2, 0, 4) == 2);
  assert(countDiceGameRounds(5, 0, 5) == 0);
  print('All tests passed!');
}