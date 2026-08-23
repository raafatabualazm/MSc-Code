@pragma('vm:entry-point')
int satelliteRelayCellScore(List<int> windows, int maxDrift) {
  int score = 0;
  for (int i = 0; i + 3 < windows.length; i += 4) {
    int ax1 = windows[i], ay1 = windows[i + 1];
    int ax2 = windows[i + 2], ay2 = windows[i + 3];
    if (ax1 > ax2) {
      int t = ax1;
      ax1 = ax2;
      ax2 = t;
    }
    if (ay1 > ay2) {
      int t = ay1;
      ay1 = ay2;
      ay2 = t;
    }
    if (ax1 == ax2 || ay1 == ay2) continue;
    for (int j = i + 4; j + 3 < windows.length; j += 4) {
      int bx1 = windows[j], by1 = windows[j + 1];
      int bx2 = windows[j + 2], by2 = windows[j + 3];
      if (bx1 > bx2) {
        int t = bx1;
        bx1 = bx2;
        bx2 = t;
      }
      if (by1 > by2) {
        int t = by1;
        by1 = by2;
        by2 = t;
      }
      if (bx1 == bx2 || by1 == by2) continue;
      int drift = (ax1 - bx1).abs() + (ay1 - by1).abs();
      if (drift > maxDrift) continue;
      int ox1 = ax1 > bx1 ? ax1 : bx1;
      int oy1 = ay1 > by1 ? ay1 : by1;
      int ox2 = ax2 < bx2 ? ax2 : bx2;
      int oy2 = ay2 < by2 ? ay2 : by2;
      if (ox1 >= ox2 || oy1 >= oy2) {
        if (ox1 == ox2 && oy1 == oy2) score += 1;
        continue;
      }
      for (int x = ox1; x < ox2; x++) {
        for (int y = oy1; y < oy2; y++) {
          int d1 = (x - ax1).abs() + (y - ay1).abs();
          int d2 = (x - bx1).abs() + (y - by1).abs();
          if (d1 + d2 <= maxDrift * 2) {
            score += 2;
          } else if (((x + y) & 1) == 0) {
            score += 1;
          }
        }
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(satelliteRelayCellScore([], 3) == 0);
  assert(satelliteRelayCellScore([0, 0, 2, 2, 1, 1, 3, 3], 3) == 2);
  assert(satelliteRelayCellScore([0, 0, 1, 1, 1, 1, 2, 2], 2) == 1);
  print('All tests passed!');
}