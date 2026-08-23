@pragma('vm:entry-point')
List<int> countHashBucketFrequencies(List<List<int>> grid) {
  var counts = <int>[];
  for (final row in grid) {
    for (final val in row) {
      if (val < 0) continue;
      if (val >= counts.length) counts.addAll(List.filled(val - counts.length + 1, 0));
      counts[val]++;
    }
  }
  return counts;
}

@pragma('vm:entry-point')
void main() {
  assert(countHashBucketFrequencies([[0]]).toString() == '[1]');
  assert(countHashBucketFrequencies([[1, 2], [0, -1]]).toString() == '[1, 1, 1]');
  assert(countHashBucketFrequencies([[], [3, 0]]).toString() == '[1, 0, 0, 1]');
  print('All tests passed!');
}