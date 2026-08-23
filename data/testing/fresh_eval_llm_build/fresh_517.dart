@pragma('vm:entry-point')
int maximizeCertifiedRecounts(List<int> tallies) {
  if (tallies.isEmpty) {
    return 0;
  }
  int n = tallies.length;
  List<int> dp = List.filled(n + 1, -1000000);
  dp[0] = 0;
  for (int i = 0; i < n; i++) {
    if (dp[i] < 0) {
      continue;
    }
    if (dp[i] > dp[i + 1]) {
      dp[i + 1] = dp[i];
    }
    int sum = 0;
    for (int j = i; j < n; j++) {
      sum += tallies[j];
      int len = j - i + 1;
      if (tallies[j] < 0 && sum <= 0) {
        continue;
      }
      if (sum > len && tallies[j] >= 0) {
        int candidate = dp[i] + 1;
        if (candidate > dp[j + 1]) {
          dp[j + 1] = candidate;
        }
      }
    }
  }
  return dp[n] < 0 ? 0 : dp[n];
}

@pragma('vm:entry-point')
void main() {
  assert(maximizeCertifiedRecounts([]) == 0);
  assert(maximizeCertifiedRecounts([3, -1, 3]) == 2);
  assert(maximizeCertifiedRecounts([2, 2, -3, 2, 2]) == 4);
  print('All tests passed!');
}