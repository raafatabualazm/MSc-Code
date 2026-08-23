@pragma('vm:entry-point')
bool isPlaylistGenreBalanced(List<String> tracks) {
  final Map<String, int> freq = {};
  for (final t in tracks) {
    freq[t] = (freq[t] ?? 0) + 1;
  }
  int maxFreq = 0;
  for (final count in freq.values) {
    if (count >= 2 && count > maxFreq) maxFreq = count;
  }
  if (maxFreq == 0) return true;
  for (final entry in freq.entries) {
    if (entry.value >= 2) {
      if (entry.value * 2 < maxFreq) return false;
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isPlaylistGenreBalanced([]) == true);
  assert(isPlaylistGenreBalanced(['pop','pop','pop','pop','pop','rock','rock','jazz']) == false);
  assert(isPlaylistGenreBalanced(['pop','pop','rock','rock','jazz']) == true);
  print('All tests passed!');
}