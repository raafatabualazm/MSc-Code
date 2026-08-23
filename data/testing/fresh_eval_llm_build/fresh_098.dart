@pragma('vm:entry-point')
String partitionPlaylistTitles(List<String> tracks) {
  var seen = <String>{};
  var shortTitles = <String>[];
  var longTitles = <String>[];
  for (var track in tracks) {
    if (seen.add(track)) {
      if (track.length <= 4) {
        shortTitles.add(track);
      } else {
        longTitles.add(track);
      }
    }
  }
  return 'S:${shortTitles.join('|')};L:${longTitles.join('|')}';
}

@pragma('vm:entry-point')
void main() {
  assert(partitionPlaylistTitles([]) == 'S:;L:');
  assert(partitionPlaylistTitles(['Muse', 'Pulse', 'Muse']) == 'S:Muse;L:Pulse');
  assert(partitionPlaylistTitles(['ABBA', 'Queen', 'ABBA', 'Kiss']) == 'S:ABBA|Kiss;L:Queen');
  print('All tests passed!');
}