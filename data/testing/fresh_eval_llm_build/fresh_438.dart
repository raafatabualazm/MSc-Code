@pragma('vm:entry-point')
bool hasCanonicalServerLogOrder(List<String> logs) {
  var expected = List<String>.from(logs);
  const rank = {'E': 0, 'W': 1, 'I': 2, 'D': 3};
  expected.sort((a, b) {
    var levelDiff = rank[a[0]]! - rank[b[0]]!;
    if (levelDiff != 0) return levelDiff;
    return int.parse(a.substring(2)) - int.parse(b.substring(2));
  });
  for (var i = 0; i < logs.length; i++) {
    if (logs[i] != expected[i]) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(hasCanonicalServerLogOrder([]) == true);
  assert(hasCanonicalServerLogOrder(['E-1', 'W-1', 'I-1']) == true);
  assert(hasCanonicalServerLogOrder(['W-1', 'E-1']) == false);
  print('All tests passed!');
}