@pragma('vm:entry-point')
int countTimetableDiamondTouches(List<int> minutes) {
  int total = 0;
  for (int i = 0; i < minutes.length; i++) {
    for (int j = 0; j < i; j++) {
      int d = (i - j) + (minutes[i] - minutes[j]).abs();
      if (d <= 3) {
        total += 4 - d;
        if (minutes[i] == minutes[j]) {
          total += 1;
        } else if ((minutes[i] - minutes[j]).abs() == 1) {
          total -= 1;
        }
      } else if (d % 2 == 0) {
        total += 1;
      }
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(countTimetableDiamondTouches([]) == 0);
  assert(countTimetableDiamondTouches([5, 5]) == 4);
  assert(countTimetableDiamondTouches([0, 2, 0, 2]) == 9);
  print('All tests passed!');
}