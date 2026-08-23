@pragma('vm:entry-point')
String classifyChargeRecovery(List<int> deltas) {
  if (deltas.isEmpty) return 'critical';
  var dp = List.filled(deltas.length, 0);
  var best = 0;
  for (var i = 0; i < deltas.length; i++) {
    dp[i] = deltas[i] > 0 ? (i == 0 ? 1 : dp[i - 1] + 1) : 0;
    if (dp[i] > best) best = dp[i];
  }
  return best == 0 ? 'critical' : (best == 1 ? 'steady' : (best < 4 ? 'buffered' : 'resilient'));
}

@pragma('vm:entry-point')
void main() {
  assert(classifyChargeRecovery([]) == 'critical');
  assert(classifyChargeRecovery([2, -1, 3, 4]) == 'buffered');
  assert(classifyChargeRecovery([1, 1, 1, 1]) == 'resilient');
  print('All tests passed!');
}