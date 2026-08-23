@pragma('vm:entry-point')
List<String> bucketCoverageZones(List<int> positions) {
  if (positions.isEmpty) return [];
  List<String> result = [];
  for (int i = 0; i < positions.length; i++) {
    String left = i == 0 ? '*' : '${(positions[i - 1] + positions[i]) ~/ 2}';
    String right = i == positions.length - 1 ? '*' : '${(positions[i] + positions[i + 1]) ~/ 2}';
    result.add('[$left,$right)');
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(bucketCoverageZones([]).toString() == '[]');
  assert(bucketCoverageZones([3]).toString() == '[[*,*)]');
  assert(bucketCoverageZones([0, 4, 10]).toString() == '[[*,2), [2,7), [7,*)]');
  print('All tests passed!');
}