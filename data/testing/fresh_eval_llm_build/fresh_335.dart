@pragma('vm:entry-point')
int measureMazeCellCooldownDebt(List<int> dailyCells, int cooldownDays) {
  int debt = 0;
  for (int day = 0; day < dailyCells.length; day++) {
    bool repeated = false;
    for (int prev = day - 1;
        prev >= 0 && day - prev <= cooldownDays;
        prev--) {
      int gap = day - prev;
      if (dailyCells[prev] == dailyCells[day]) {
        repeated = true;
        if (gap == 1) {
          debt += 4;
        } else if (gap == cooldownDays) {
          debt += 1;
        } else {
          debt += 2;
        }
        break;
      }
    }
    if (!repeated && day >= cooldownDays && cooldownDays > 0) {
      debt -= 1;
    }
  }
  return debt;
}

@pragma('vm:entry-point')
void main() {
  assert(measureMazeCellCooldownDebt([], 3) == 0);
  assert(measureMazeCellCooldownDebt([1, 1], 2) == 4);
  assert(measureMazeCellCooldownDebt([1, 2, 3, 4], 2) == -2);
  print('All tests passed!');
}