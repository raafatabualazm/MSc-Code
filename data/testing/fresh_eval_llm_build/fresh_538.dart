@pragma('vm:entry-point')
List<int> analyzeRgbHotspotGeometry(List<int> pixels, int width, int threshold) {
  if (width <= 0 || pixels.isEmpty || pixels.length % (width * 3) != 0) {
    return [0, 0, 0];
  }
  int height = pixels.length ~/ (width * 3);
  int active = 0, minX = width, minY = height, maxX = -1, maxY = -1, border = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int i = (y * width + x) * 3;
      int r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
      int hi = r > g ? (r > b ? r : b) : (g > b ? g : b);
      int lo = r < g ? (r < b ? r : b) : (g < b ? g : b);
      if (hi - lo < threshold) continue;
      active++;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      for (int d = 0; d < 2; d++) {
        int nx = d == 0 ? x + 1 : x;
        int ny = d == 0 ? y : y + 1;
        if (nx >= width || ny >= height) {
          border++;
          continue;
        }
        int n = (ny * width + nx) * 3;
        int nr = pixels[n], ng = pixels[n + 1], nb = pixels[n + 2];
        int nhi = nr > ng ? (nr > nb ? nr : nb) : (ng > nb ? ng : nb);
        int nlo = nr < ng ? (nr < nb ? nr : nb) : (ng < nb ? ng : nb);
        if (nhi - nlo < threshold) border++;
      }
    }
  }
  if (active == 0) return [0, 0, border];
  return [active, (maxX - minX + 1) * (maxY - minY + 1), border];
}

@pragma('vm:entry-point')
void main() {
  assert(analyzeRgbHotspotGeometry([], 1, 3).toString() == '[0, 0, 0]');
  assert(analyzeRgbHotspotGeometry([255, 0, 0], 1, 5).toString() == '[1, 1, 2]');
  assert(analyzeRgbHotspotGeometry([255, 0, 0, 1, 1, 1, 0, 255, 0], 3, 10).toString() == '[2, 3, 4]');
  print('All tests passed!');
}