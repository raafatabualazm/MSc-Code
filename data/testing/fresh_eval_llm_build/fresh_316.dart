@pragma('vm:entry-point')
int wifiBinRelayScore(List<int> bins) {
  if (bins.isEmpty) return 0;
  List<int> dp = List.filled(bins.length, 1);
  int best = 1;
  for (int i = 0; i < bins.length; i++) {
    for (int j = 0; j < i; j++) {
      int diff = (bins[i] - bins[j]).abs();
      if (diff == 0) {
        int cand = dp[j] + 2;
        if (cand > dp[i]) dp[i] = cand;
      } else if (diff == 1) {
        int cand = dp[j] + 1;
        if (cand > dp[i]) dp[i] = cand;
      } else if (diff >= 4 && bins[i] > bins[j] && dp[i] < 2) {
        dp[i] = 2;
      }
    }
    if (dp[i] > best) best = dp[i];
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(wifiBinRelayScore([]) == 0);
  assert(wifiBinRelayScore([2, 2, 1, 1]) == 6);
  assert(wifiBinRelayScore([1, 4]) == 1);
  print('All tests passed!');
}