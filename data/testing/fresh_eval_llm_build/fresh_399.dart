@pragma('vm:entry-point')
List<List<int>> recipeJarCarryGrid(List<int> spoonUnits, int days) {
  if (days <= 0) {
    return List.generate(spoonUnits.length, (_) => <int>[]);
  }
  final dp = <List<int>>[];
  for (var i = 0; i < spoonUnits.length; i++) {
    final row = <int>[];
    var carry = 0;
    for (var d = 1; d <= days; d++) {
      var scaled = spoonUnits[i] * d + carry;
      if (scaled % 5 == 0) {
        carry = 1;
        scaled -= 2;
      } else if (scaled.isOdd) {
        carry = 2;
        scaled += 1;
      } else {
        carry = 0;
      }
      if (i > 0 && scaled < dp[i - 1][d - 1]) {
        scaled += 1;
      }
      row.add(scaled);
    }
    dp.add(row);
  }
  return dp;
}

@pragma('vm:entry-point')
void main() {
  assert(recipeJarCarryGrid([2], 3).toString() == '[[2, 4, 6]]');
  assert(recipeJarCarryGrid([5], 2).toString() == '[[3, 12]]');
  assert(recipeJarCarryGrid([], 4).toString() == '[]');
  print('All tests passed!');
}