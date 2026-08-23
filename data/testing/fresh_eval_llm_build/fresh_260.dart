@pragma('vm:entry-point')
num warehousePrimeBatchScore(List<int> itemCounts, int rackBase) {
  int base = rackBase < 0 ? -rackBase : rackBase;
  if (base == 0) {
    return 0;
  }
  int score = 0;
  for (int count in itemCounts) {
    int value = count < 0 ? -count : count;
    bool prime = value >= 2;
    for (int d = 2; d * d <= value; d++) {
      if (value % d == 0) {
        prime = false;
        break;
      }
    }
    if (prime) {
      score += value % base;
    } else if (value > 0) {
      int a = value, b = base;
      while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
      }
      score -= a;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(warehousePrimeBatchScore([2, 3, 4], 5) == 4);
  assert(warehousePrimeBatchScore([1, 0, -3, 8], 6) == 0);
  assert(warehousePrimeBatchScore([25], 5) == -5);
  print('All tests passed!');
}