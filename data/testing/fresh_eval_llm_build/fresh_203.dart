@pragma('vm:entry-point')
int countServerLogOverlapScore(List<String> logs, int minArea) {
  List<List<int>> rects = [];
  for (var line in logs) {
    var p = line.split(',');
    if (p.length != 4) continue;
    int x1 = int.parse(p[0]), y1 = int.parse(p[1]);
    int x2 = int.parse(p[2]), y2 = int.parse(p[3]);
    if (x1 == x2 || y1 == y2) continue;
    if (x1 > x2) { var t = x1; x1 = x2; x2 = t; }
    if (y1 > y2) { var t = y1; y1 = y2; y2 = t; }
    rects.add([x1, y1, x2, y2]);
  }
  int score = 0;
  for (int i = 0; i < rects.length; i++) {
    for (int j = i + 1; j < rects.length; j++) {
      var a = rects[i], b = rects[j];
      int w = (a[2] < b[2] ? a[2] : b[2]) - (a[0] > b[0] ? a[0] : b[0]);
      int h = (a[3] < b[3] ? a[3] : b[3]) - (a[1] > b[1] ? a[1] : b[1]);
      int area = (w > 0 && h > 0) ? w * h : 0;
      if (area >= minArea) {
        int areaA = (a[2] - a[0]) * (a[3] - a[1]);
        int areaB = (b[2] - b[0]) * (b[3] - b[1]);
        score += area + ((area == areaA || area == areaB) ? 1 : 0);
        continue;
      }
      int dx = ((a[0] > b[0] ? a[0] : b[0]) - (a[2] < b[2] ? a[2] : b[2]));
      int dy = ((a[1] > b[1] ? a[1] : b[1]) - (a[3] < b[3] ? a[3] : b[3]));
      if (area == 0 && (dx > 0 ? dx : 0) + (dy > 0 ? dy : 0) == 1) score--;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(countServerLogOverlapScore([], 1) == 0);
  assert(countServerLogOverlapScore(['0,0,3,3','1,1,4,4'], 1) == 4);
  assert(countServerLogOverlapScore(['0,0,2,2','3,0,5,2'], 1) == -1);
  print('All tests passed!');
}