@pragma('vm:entry-point')
int tripleRoundFaceScore(List<int> rounds) {
  final counts = <int, int>{};
  var score = 0;
  for (final face in rounds) {
    final next = (counts[face] ?? 0) + 1;
    counts[face] = next;
    if (next == 3) {
      score += face;
    } else if (next == 4) {
      score -= face;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(tripleRoundFaceScore([]) == 0);
  assert(tripleRoundFaceScore([2, 2, 2]) == 2);
  assert(tripleRoundFaceScore([1, 1, 1, 1, 6, 6, 6]) == 6);
  print('All tests passed!');
}