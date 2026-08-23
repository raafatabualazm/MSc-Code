@pragma('vm:entry-point')
int countExpressMinutePairings(List<int> minutes) {
  int solve(int left, int right) {
    if (left > right) return 1;
    int total = 0;
    if (minutes[left] % 15 == 0) {
      total += solve(left + 1, right);
    }
    for (int i = left + 1; i <= right; i++) {
      int gap = (minutes[i] - minutes[left]).abs();
      if ((minutes[left] + minutes[i]) % 10 == 0) {
        if (gap <= 20) {
          total += solve(left + 1, i - 1) * solve(i + 1, right);
        }
      }
    }
    return total;
  }

  return solve(0, minutes.length - 1);
}

@pragma('vm:entry-point')
void main() {
  assert(countExpressMinutePairings([]) == 1);
  assert(countExpressMinutePairings([10, 20]) == 1);
  assert(countExpressMinutePairings([15, 15, 15]) == 4);
  print('All tests passed!');
}