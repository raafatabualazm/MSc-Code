@pragma('vm:entry-point')
import 'dart:math';

String mapTimetableMinuteCrowding(List<String> zones) {
  if (zones.isEmpty) return 'empty';
  int score = 0, overlaps = 0;
  for (int i = 0; i < zones.length; i++) {
    var a = zones[i].split(',');
    int ma = int.parse(a[0]), ax1 = int.parse(a[1]), ay1 = int.parse(a[2]), ax2 = int.parse(a[3]), ay2 = int.parse(a[4]);
    if (ax1 == ax2 || ay1 == ay2) {
      score--;
      continue;
    }
    for (int j = i + 1; j < zones.length; j++) {
      var b = zones[j].split(',');
      int mb = int.parse(b[0]), bx1 = int.parse(b[1]), by1 = int.parse(b[2]), bx2 = int.parse(b[3]), by2 = int.parse(b[4]);
      int dx = min(ax2, bx2) - max(ax1, bx1), dy = min(ay2, by2) - max(ay1, by1);
      if (dx > 0 && dy > 0) {
        int area = dx * dy;
        overlaps++;
        if (ma == mb) score += area * 2;
        else if ((ma - mb).abs() == 1) score += area;
        else score -= area;
      } else {
        int gap = (ax1 - bx1).abs() + (ay1 - by1).abs();
        if (gap == 0) continue;
        score += gap.isEven ? 1 : -1;
      }
    }
  }
  if (overlaps == 0 && score <= 0) return 'quiet:$score';
  if (overlaps > zones.length ~/ 2) return 'packed:$score';
  return 'mixed:$score';
}

@pragma('vm:entry-point')
void main() {
  assert(mapTimetableMinuteCrowding([]) == 'empty');
  assert(mapTimetableMinuteCrowding(['5,0,0,2,2','5,1,1,3,3']) == 'mixed:2');
  assert(mapTimetableMinuteCrowding(['1,0,0,0,2']) == 'quiet:-1');
  print('All tests passed!');
}