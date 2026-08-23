@pragma('vm:entry-point')
int indexOfFirstNearMissSpelling(List<String> spellCheckWords) {
  if (spellCheckWords.isEmpty) return -1;
  String target = spellCheckWords[0];
  int targetLen = target.length;
  List<String> suggestions = spellCheckWords.sublist(1);
  suggestions.sort((a, b) =>
      (a.length - targetLen).abs().compareTo((b.length - targetLen).abs()));
  for (int i = 0; i < suggestions.length; i++) {
    if ((suggestions[i].length - targetLen).abs() == 1) {
      return i;
    }
  }
  return -1;
}

@pragma('vm:entry-point')
void main() {
  assert(indexOfFirstNearMissSpelling([]) == -1);
  assert(indexOfFirstNearMissSpelling(["hello"]) == -1);
  assert(indexOfFirstNearMissSpelling(["cat", "dog", "bat", "cats", "cut"]) == 3);
  print('All tests passed!');
}