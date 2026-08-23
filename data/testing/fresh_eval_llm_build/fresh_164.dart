@pragma('vm:entry-point')
int shelfCodeIntervalPenalty(List<int> shelfDays) {
  int total = 0;
  for (int i = 1; i < shelfDays.length; i++) {
    int diff = (shelfDays[i] - shelfDays[i - 1]).abs();
    if (diff == 0) {
      total += 4;
    } else {
      total += (diff ~/ 7) * 2;
      int rem = diff % 7;
      if (rem == 0) {
        total += 1;
      } else if (rem <= 2) {
        total += rem;
      } else {
        total += rem + 1;
      }
      if ((shelfDays[i] < 0) != (shelfDays[i - 1] < 0)) {
        total -= 1;
      }
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(shelfCodeIntervalPenalty([]) == 0);
  assert(shelfCodeIntervalPenalty([1, 1]) == 4);
  assert(shelfCodeIntervalPenalty([-5, 5, 12]) == 8);
  print('All tests passed!');
}