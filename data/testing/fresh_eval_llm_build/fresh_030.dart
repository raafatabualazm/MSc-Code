@pragma('vm:entry-point')
List<int> spellCheckDeletionIndices(String word) {
  List<int> result = [];
  int run = 0;
  for (int i = 0; i < word.length; i++) {
    if (i == 0 || word[i] != word[i - 1]) {
      run = 1;
    } else {
      run++;
      if (run > 2) {
        result.add(i);
      }
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(spellCheckDeletionIndices('').toString() == '[]');
  assert(spellCheckDeletionIndices('hello').toString() == '[]');
  assert(spellCheckDeletionIndices('aaabbb').toString() == '[2, 5]');
  print('All tests passed!');
}