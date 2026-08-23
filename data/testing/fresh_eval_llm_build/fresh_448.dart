@pragma('vm:entry-point')
int medianEliminationRoundIndex(List<int> eliminationCounts) {
  if (eliminationCounts.isEmpty) return -1;
  int total = 0;
  for (int e in eliminationCounts) {
    total += e;
  }
  if (total == 0) return -1;
  int target = (total + 1) ~/ 2;
  int left = 0;
  int right = eliminationCounts.length - 1;
  while (left < right) {
    int mid = (left + right) ~/ 2;
    int sum = 0;
    for (int i = 0; i <= mid; i++) {
      sum += eliminationCounts[i];
    }
    if (sum < target) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

@pragma('vm:entry-point')
void main() {
  assert(medianEliminationRoundIndex([1, 2, 3]) == 1);
  assert(medianEliminationRoundIndex([]) == -1);
  assert(medianEliminationRoundIndex([5]) == 0);
  print('All tests passed!');
}