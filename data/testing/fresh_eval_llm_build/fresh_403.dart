@pragma('vm:entry-point')
List<String> stabilizeSpellEdits(List<String> words) {
  if (words.isEmpty) return [];
  List<String> out = [words[0]];
  for (int k = 1; k < words.length; k++) {
    String a = out.last, b = words[k];
    List<List<int>> dp = List.generate(a.length + 1, (i) => List.filled(b.length + 1, 0));
    for (int i = 0; i <= a.length; i++) dp[i][0] = i;
    for (int j = 0; j <= b.length; j++) dp[0][j] = j;
    for (int i = 1; i <= a.length; i++) {
      for (int j = 1; j <= b.length; j++) {
        int best = dp[i - 1][j] + 1;
        if (dp[i][j - 1] + 1 < best) best = dp[i][j - 1] + 1;
        int diag = dp[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
        dp[i][j] = diag < best ? diag : best;
      }
    }
    out.add(dp[a.length][b.length] <= 1 ? a : b);
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(stabilizeSpellEdits([]).isEmpty);
  assert(stabilizeSpellEdits(['book', 'books']).toString() == '[book, book]');
  assert(stabilizeSpellEdits(['code', 'coda', 'cola']).toString() == '[code, code, cola]');
  print('All tests passed!');
}