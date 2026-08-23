@pragma('vm:entry-point')
bool canSplitDiceRoundsWithAverageGuard(List<int> scores) {
  int n = scores.length;
  if (n < 2) return false;
  int totalSum = 0;
  for (int v in scores) totalSum += v;
  int prefixSum = 0;
  for (int k = 1; k < n; k++) {
    prefixSum += scores[k - 1];
    int suffixSum = totalSum - prefixSum;
    if (prefixSum > suffixSum) {
      int maxSuffix = 0;
      for (int j = k; j < n; j++) {
        if (scores[j] > maxSuffix) maxSuffix = scores[j];
      }
      if (maxSuffix * k <= prefixSum) return true;
    }
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(canSplitDiceRoundsWithAverageGuard([6, 5, 4, 3]) == true);
  assert(canSplitDiceRoundsWithAverageGuard([1, 2, 3, 4]) == false);
  assert(canSplitDiceRoundsWithAverageGuard([5, 2, 2]) == true);
  print('All tests passed!');
}