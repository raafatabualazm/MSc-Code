@pragma('vm:entry-point')
List<String> flaggedShelfRuns(String codes, int targetRun) {
  if (codes.isEmpty || targetRun <= 0) return [];
  final dp = List<int>.filled(codes.length, 1);
  final out = <String>[];
  for (var i = 0; i < codes.length; i++) {
    if (i > 0 && codes[i] == codes[i - 1]) dp[i] = dp[i - 1] + 1;
    if (dp[i] >= targetRun) {
      out.add(codes.substring(i - targetRun + 1, i + 1));
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(flaggedShelfRuns('AABBCC', 2).toString() == '[AA, BB, CC]');
  assert(flaggedShelfRuns('', 3).length == 0);
  assert(flaggedShelfRuns('ZZZZ', 3).toString() == '[ZZZ, ZZZ]');
  print('All tests passed!');
}