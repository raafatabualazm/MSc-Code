@pragma('vm:entry-point')
String? classifyQrAnchorPatch(List<String> modules) {
  if (modules.isEmpty || modules[0].isEmpty) return null;
  int w = modules[0].length;
  for (final row in modules) {
    if (row.length != w) return null;
  }
  List<List<int>> dp = List.generate(modules.length, (_) => List.filled(w, 0));
  int best = 0;
  int edgeHits = 0;
  for (int i = 0; i < modules.length; i++) {
    for (int j = 0; j < w; j++) {
      if (modules[i][j] == '#') {
        dp[i][j] = (i == 0 || j == 0) ? 1 : 1 + [dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]].reduce((a, b) => a < b ? a : b);
        bool edge = i - dp[i][j] + 1 == 0 || j - dp[i][j] + 1 == 0 || i == modules.length - 1 || j == w - 1;
        if (dp[i][j] > best) {
          best = dp[i][j];
          edgeHits = edge ? 1 : 0;
        } else if (dp[i][j] == best && edge) {
          edgeHits++;
        }
      }
    }
  }
  if (best == 0) return "blank";
  if (best == 1) return edgeHits > 2 ? "speckled" : "isolated";
  return edgeHits >= 2 ? "anchored" : "floating";
}

@pragma('vm:entry-point')
void main() {
  assert(classifyQrAnchorPatch([]) == null);
  assert(classifyQrAnchorPatch(['....','.##.','.##.','....']) == 'floating');
  assert(classifyQrAnchorPatch(['##..','##..','..##','..##']) == 'anchored');
  print('All tests passed!');
}