@pragma('vm:entry-point')
int rankedPositiveTideWeight(List<int> readings) {
  var sorted = List<int>.from(readings);
  sorted.sort((a, b) {
    var diff = a.abs() - b.abs();
    return diff != 0 ? diff : b - a;
  });
  int score = 0;
  for (int i = 0; i < sorted.length; i++) {
    if (sorted[i] >= 0) score += i + 1;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(rankedPositiveTideWeight([]) == 0);
  assert(rankedPositiveTideWeight([-1, 1, 2]) == 4);
  assert(rankedPositiveTideWeight([5, -5, 0]) == 3);
  print('All tests passed!');
}