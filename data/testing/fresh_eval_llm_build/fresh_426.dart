@pragma('vm:entry-point')
int countSongsWithMatchingLengthAndFrequency(List<String> songs) {
  var freq = <String, int>{};
  var matching = <String>{};
  for (var song in songs) {
    int old = freq[song] ?? 0;
    int next = old + 1;
    freq[song] = next;
    if (old == song.length) {
      matching.remove(song);
    } else if (next == song.length) {
      matching.add(song);
    }
  }
  return matching.length;
}

@pragma('vm:entry-point')
void main() {
  assert(countSongsWithMatchingLengthAndFrequency([]) == 0);
  assert(countSongsWithMatchingLengthAndFrequency(["a","ab","ab"]) == 2);
  assert(countSongsWithMatchingLengthAndFrequency(["ab","ab","ab"]) == 0);
  print('All tests passed!');
}