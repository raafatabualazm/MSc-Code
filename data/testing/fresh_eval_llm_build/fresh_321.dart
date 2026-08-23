@pragma('vm:entry-point')
bool isSortedByLengthThenLexico(List<String> candidates) {
  if (candidates.length <= 2) return true;
  int targetLen = candidates[0].length;
  for (int i = 1; i < candidates.length - 1; i++) {
    String a = candidates[i];
    String b = candidates[i + 1];
    int diffA = (a.length - targetLen).abs();
    int diffB = (b.length - targetLen).abs();
    if (diffA < diffB) continue;
    if (diffA > diffB) return false;
    if (a.compareTo(b) > 0) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isSortedByLengthThenLexico([]));
  assert(!isSortedByLengthThenLexico(['hi', 'hello', 'hey']));
  assert(isSortedByLengthThenLexico(['cat', 'ant', 'bee', 'cow']));
  print('All tests passed!');
}