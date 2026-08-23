@pragma('vm:entry-point')
int scoreSatellitePassQueue(List<String> windows) {
  var items = List<String>.from(windows);
  items.sort((a, b) {
    var pa = a.split('|');
    var pb = b.split('|');
    int da = int.parse(pa[2]) - int.parse(pa[1]);
    int db = int.parse(pb[2]) - int.parse(pb[1]);
    if (da != db) return db - da;
    int sa = int.parse(pa[1]);
    int sb = int.parse(pb[1]);
    if (sa != sb) return sa - sb;
    return pa[0].compareTo(pb[0]);
  });
  int score = 0;
  int previousEnd = -1;
  for (var item in items) {
    var p = item.split('|');
    int start = int.parse(p[1]);
    int end = int.parse(p[2]);
    int duration = end - start;
    if (previousEnd >= 0) {
      int gap = start - previousEnd;
      if (gap < 0) {
        score += duration;
      } else if (gap <= 2) {
        score += gap + 1;
      } else {
        score -= 1;
      }
    } else if (duration >= 5) {
      score += 2;
    } else {
      score += 1;
    }
    previousEnd = end;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(scoreSatellitePassQueue([]) == 0);
  assert(scoreSatellitePassQueue(['A|0|5']) == 2);
  assert(scoreSatellitePassQueue(['A|0|5','B|9|12']) == 1);
  print('All tests passed!');
}