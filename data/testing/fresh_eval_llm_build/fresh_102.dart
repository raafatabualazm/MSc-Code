@pragma('vm:entry-point')
List<double> shelfCodeCarryover(List<int> codes) {
  if (codes.isEmpty) return [];
  List<double> dp = List.filled(codes.length, 0.0);
  dp[0] = codes[0].isEven ? 1.0 : -1.0;
  for (int i = 1; i < codes.length; i++) {
    dp[i] = dp[i - 1] + ((codes[i] - codes[i - 1]).abs() <= 2 ? 0.5 : -0.5);
  }
  return dp;
}

@pragma('vm:entry-point')
void main() {
  assert(shelfCodeCarryover([]).toString() == '[]');
  assert(shelfCodeCarryover([4]).toString() == '[1.0]');
  assert(shelfCodeCarryover([3, 5, 8]).toString() == '[-1.0, -0.5, -1.0]');
  print('All tests passed!');
}