@pragma('vm:entry-point')
int countQuietViewingSections(List<int> blockedPairsByRow, int maxBlocked) {
  int streak = 0;
  int total = 0;
  for (final blocked in blockedPairsByRow) {
    streak = blocked <= maxBlocked ? streak + 1 : 0;
    total += streak;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(countQuietViewingSections([], 2) == 0);
  assert(countQuietViewingSections([1, 1, 2, 1], 1) == 4);
  assert(countQuietViewingSections([3, 3, 3], 3) == 6);
  print('All tests passed!');
}