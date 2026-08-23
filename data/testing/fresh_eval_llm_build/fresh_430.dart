@pragma('vm:entry-point')
int countCardSkipsToDay(List<int> deck, int targetDay) {
  int day = 0;
  int count = 0;
  if (day >= targetDay) return count;
  for (int c in deck) {
    day += c;
    count++;
    if (day >= targetDay) return count;
  }
  return -1;
}

@pragma('vm:entry-point')
void main() {
  assert(countCardSkipsToDay([5, 10, 15], 0) == 0);
  assert(countCardSkipsToDay([5, 10, 15], 15) == 2);
  assert(countCardSkipsToDay([], 10) == -1);
  print('All tests passed!');
}