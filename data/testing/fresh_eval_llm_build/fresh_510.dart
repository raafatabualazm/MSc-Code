@pragma('vm:entry-point')
List<int> analyzeRgbBoundingMix(List<int> pixels) {
  int rMinX = 1 << 30, rMinY = 1 << 30, rMaxX = -(1 << 30), rMaxY = -(1 << 30);
  int bMinX = 1 << 30, bMinY = 1 << 30, bMaxX = -(1 << 30), bMaxY = -(1 << 30);
  List<int> reds = [], blues = [];
  for (int i = 0; i + 4 < pixels.length; i += 5) {
    int x = pixels[i], y = pixels[i + 1], r = pixels[i + 2], g = pixels[i + 3], b = pixels[i + 4];
    if (r > g && r > b) {
      if (x < rMinX) rMinX = x; if (y < rMinY) rMinY = y; if (x > rMaxX) rMaxX = x; if (y > rMaxY) rMaxY = y; reds..add(x)..add(y);
    } else if (b > r && b > g) {
      if (x < bMinX) bMinX = x; if (y < bMinY) bMinY = y; if (x > bMaxX) bMaxX = x; if (y > bMaxY) bMaxY = y; blues..add(x)..add(y);
    }
  }
  int rArea = reds.isEmpty ? 0 : (rMaxX - rMinX + 1) * (rMaxY - rMinY + 1), bArea = blues.isEmpty ? 0 : (bMaxX - bMinX + 1) * (bMaxY - bMinY + 1);
  int ox = reds.isEmpty || blues.isEmpty ? 0 : ((rMaxX < bMaxX ? rMaxX : bMaxX) - (rMinX > bMinX ? rMinX : bMinX) + 1), oy = reds.isEmpty || blues.isEmpty ? 0 : ((rMaxY < bMaxY ? rMaxY : bMaxY) - (rMinY > bMinY ? rMinY : bMinY) + 1);
  int best = -1;
  for (int i = 0; i < reds.length; i += 2) {
    for (int j = 0; j < blues.length; j += 2) {
      int d = (reds[i] - blues[j]).abs() + (reds[i + 1] - blues[j + 1]).abs();
      if (best == -1 || d < best) best = d;
    }
  }
  return [rArea, bArea, (ox > 0 && oy > 0) ? ox * oy : 0, best];
}

@pragma('vm:entry-point')
void main() {
  assert(analyzeRgbBoundingMix([]).toString() == '[0, 0, 0, -1]');
  assert(analyzeRgbBoundingMix([0,0,9,1,2,2,1,1,2,9]).toString() == '[1, 1, 0, 3]');
  assert(analyzeRgbBoundingMix([0,0,9,1,2,2,2,8,0,1,1,1,0,2,9,3,3,1,0,7]).toString() == '[9, 9, 4, 2]');
  print('All tests passed!');
}