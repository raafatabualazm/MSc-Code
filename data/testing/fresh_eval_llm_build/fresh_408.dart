@pragma('vm:entry-point')
bool bracketHasDominantEvenMargin(List<int> margins) {
  if (margins.isEmpty) return false;
  int n = margins.length;
  if (n & (n - 1) != 0) return false;
  Map<int, int> freq = {};
  for (int m in margins) {
    freq[m] = (freq[m] ?? 0) + 1;
  }
  for (var entry in freq.entries) {
    if (entry.value * 2 >= n && entry.key.isEven) {
      return true;
    }
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(bracketHasDominantEvenMargin([]) == false);
  assert(bracketHasDominantEvenMargin([2]) == true);
  assert(bracketHasDominantEvenMargin([1, 3, 5]) == false);
  print('All tests passed!');
}