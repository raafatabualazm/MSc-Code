@pragma('vm:entry-point')
bool isMorseDiffPalindrome(String transmission) {
  List<String> tokens = transmission.split(' ');
  List<int> diffs = [];
  for (var token in tokens) {
    if (token.isEmpty) return false;
    int dashes = 0, dots = 0, run = 1;
    for (int i = 0; i < token.length; i++) {
      String ch = token[i];
      if (ch != '.' && ch != '-') return false;
      if (ch == '-') dashes++; else dots++;
      if (i > 0 && ch == token[i - 1]) run++; else run = 1;
      if (run > 2) return false;
    }
    diffs.add(dashes - dots);
  }
  int n = diffs.length;
  for (int i = 0; i < n ~/ 2; i++) {
    if (diffs[i] != diffs[n - 1 - i]) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isMorseDiffPalindrome('.-') == true);
  assert(isMorseDiffPalindrome('---..') == false);
  assert(isMorseDiffPalindrome('.- .- -..') == false);
  print('All tests passed!');
}