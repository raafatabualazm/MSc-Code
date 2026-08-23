@pragma('vm:entry-point')
int countDominantArtists(List<String> songs) {
  if (songs.isEmpty) return 0;
  final artistTotals = <String, int>{};
  for (final song in songs) {
    final parts = song.split(':');
    if (parts.length != 3) continue;
    final artist = parts[0];
    final duration = int.tryParse(parts[2]);
    if (duration == null) continue;
    artistTotals.update(artist, (existing) => existing + duration, ifAbsent: () => duration);
  }
  if (artistTotals.isEmpty) return 0;
  final entries = artistTotals.entries.toList();
  final totalArtists = entries.length;
  int dominantCount = 0;
  for (final entry in entries) {
    final currentArtist = entry.key;
    final currentTotal = entry.value;
    int lessCount = 0;
    for (final other in entries) {
      if (other.key == currentArtist) continue;
      if (other.value < currentTotal) {
        lessCount++;
      }
    }
    if (lessCount >= (totalArtists ~/ 2)) {
      dominantCount++;
    }
  }
  return dominantCount;
}

@pragma('vm:entry-point')
void main() {
  assert(countDominantArtists([]) == 0);
  assert(countDominantArtists(["A:x:100"]) == 1);
  assert(countDominantArtists(["A:x:100", "B:y:100"]) == 0);
  print('All tests passed!');
}