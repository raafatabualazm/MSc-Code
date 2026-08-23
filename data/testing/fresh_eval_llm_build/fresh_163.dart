@pragma('vm:entry-point')
String dominantArtistLongTitleSummary(List<String> playlist, int threshold) {
  if (playlist.isEmpty) return 'none';
  final Map<String, int> counts = {};
  final Map<String, int> longTitles = {};
  for (final entry in playlist) {
    final colon = entry.indexOf(':');
    if (colon < 0) continue;
    final artist = entry.substring(0, colon);
    final title = entry.substring(colon + 1);
    counts[artist] = (counts[artist] ?? 0) + 1;
    if (title.length > 5) {
      longTitles[artist] = (longTitles[artist] ?? 0) + 1;
    } else {
      longTitles.putIfAbsent(artist, () => 0);
    }
  }
  final List<String> qualifying = [];
  for (final artist in counts.keys) {
    if (counts[artist]! > threshold) {
      qualifying.add(artist);
    }
  }
  if (qualifying.isEmpty) return 'none';
  qualifying.sort();
  final buffer = StringBuffer();
  for (int i = 0; i < qualifying.length; i++) {
    final artist = qualifying[i];
    final total = counts[artist]!;
    final long = longTitles[artist] ?? 0;
    if (i > 0) buffer.write('\n');
    buffer.write('$artist($total/$long)');
  }
  return buffer.toString();
}

@pragma('vm:entry-point')
void main() {
  assert(dominantArtistLongTitleSummary([], 0) == 'none');
  assert(dominantArtistLongTitleSummary(['a:hi', 'a:hi', 'a:hello world', 'b:x'], 1) == 'a(3/1)');
  assert(dominantArtistLongTitleSummary(['a:song', 'a:song', 'b:track1x', 'b:track2x', 'b:ok'], 1) == 'a(2/0)
b(3/2)');
  print('All tests passed!');
}