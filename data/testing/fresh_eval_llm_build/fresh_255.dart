@pragma('vm:entry-point')
int totalLikedDuration(String playlist) {
  int total = 0;
  for (var entry in playlist.split(';')) {
    var parts = entry.split(',');
    if (parts.length != 3) continue;
    if (parts[2].trim() != '1') continue;
    var dur = parts[1].trim();
    int? secs;
    if (dur.contains(':')) {
      var t = dur.split(':');
      if (t.length != 2) continue;
      var m = int.tryParse(t[0]);
      var s = int.tryParse(t[1]);
      if (m == null || s == null || s < 0 || s > 59) continue;
      secs = m * 60 + s;
    } else {
      secs = int.tryParse(dur);
    }
    if (secs != null && secs > 0) total += secs;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(totalLikedDuration("") == 0);
  assert(totalLikedDuration("a,45,1") == 45);
  assert(totalLikedDuration("a,1:30,1;b,45,0;c,2:00,1") == 210);
  print('All tests passed!');
}