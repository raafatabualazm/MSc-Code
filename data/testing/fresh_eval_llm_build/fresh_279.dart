@pragma('vm:entry-point')
double packetRelayConsistencyScore(List<int> packets, int tolerance) {
  if (packets.isEmpty || tolerance < 0) {
    return 0.0;
  }
  int n = packets.length;
  List<double> dp = List.filled(n, -1024.0);
  double best = 0.0;

  for (int i = 0; i < n; i++) {
    if (packets[i] <= 0) {
      continue;
    }
    dp[i] = packets[i].isEven ? 1.0 : 0.5;

    for (int j = 0; j < i; j++) {
      if (dp[j] < 0.0) {
        continue;
      }
      int diff = (packets[i] - packets[j]).abs();
      double gain;

      if (diff <= tolerance) {
        gain = 1.0;
      } else if (diff <= tolerance + 2 &&
          ((packets[i] + packets[j]) % 4 == 0)) {
        gain = 0.5;
      } else {
        continue;
      }

      double candidate = dp[j] + gain;
      if (candidate > dp[i]) {
        dp[i] = candidate;
      }
    }

    if (dp[i] > best) {
      best = dp[i];
    }
  }

  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(packetRelayConsistencyScore([2, 3], 1) == 2.0);
  assert(packetRelayConsistencyScore([5, 7], 1) == 1.0);
  assert(packetRelayConsistencyScore([], 3) == 0.0);
  print('All tests passed!');
}