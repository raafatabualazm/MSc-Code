@pragma('vm:entry-point')
int countLibraryShelfMoves(int lastMoveDay) {
  int count = 0;
  for (int d = lastMoveDay + 7; d <= 365; d += 7) {
    int weekday = (d - 1) % 7;
    if (weekday < 5) {
      count++;
    }
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(candidate(1) == 52);
  assert(candidate(6) == 0);
  assert(candidate(358) == 1);
  print('All tests passed!');
}