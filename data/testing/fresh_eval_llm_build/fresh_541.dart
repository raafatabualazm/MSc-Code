@pragma('vm:entry-point')
int layeredDnaMirrorScore(String dna) {
  int n = dna.length;
  if (n < 2) return 0;
  List<List<int>> dp = List.generate(n, (_) => List.filled(n, 0));
  int total = 0;
  for (int len = 2; len <= n; len++) {
    for (int i = 0; i + len <= n; i++) {
      int j = i + len - 1;
      bool pair = (dna[i] == 'A' && dna[j] == 'T') ||
          (dna[i] == 'T' && dna[j] == 'A') ||
          (dna[i] == 'C' && dna[j] == 'G') ||
          (dna[i] == 'G' && dna[j] == 'C');
      if (!pair) {
        continue;
      }
      if (len > 2 && dp[i + 1][j - 1] == 0) {
        continue;
      }
      dp[i][j] = 1;
      total += len;
      if (len == 2) {
        total += 1;
      } else if (dna[i] == 'C' || dna[i] == 'G') {
        total += 2;
      }
      if (len % 4 == 0) {
        total += 1;
      }
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(layeredDnaMirrorScore('AT') == 3);
  assert(layeredDnaMirrorScore('CGCG') == 16);
  assert(layeredDnaMirrorScore('AG') == 0);
  print('All tests passed!');
}