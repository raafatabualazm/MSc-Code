@pragma('vm:entry-point')
String ratePlaylistDetours(List<String> playlist) {
  int search(int i, int vibe) {
    if (i >= playlist.length) return vibe;
    String song = playlist[i];
    if (song == "?") {
      int extend = search(i + 1, vibe + 2);
      int coolDown = search(i + 1, vibe ~/ 2);
      return extend > coolDown ? extend : coolDown;
    }
    for (int j = 0; j < song.length; j++) {
      if ('aeiou'.contains(song[j])) {
        vibe += 1;
      } else if (song[j] == song[j].toUpperCase()) {
        vibe -= 1;
      }
    }
    return search(i + 1, vibe);
  }
  int score = search(0, 0);
  if (score >= 6) return 'festival';
  if (score >= 3) return 'roadtrip';
  if (score >= 0) return 'lounge';
  return 'mute';
}

@pragma('vm:entry-point')
void main() {
  assert(ratePlaylistDetours([]) == 'lounge');
  assert(ratePlaylistDetours(['aei']) == 'roadtrip');
  assert(ratePlaylistDetours(['AEI']) == 'mute');
  print('All tests passed!');
}