@pragma('vm:entry-point')
List<int> dominantRgbPersistence(List<int> pixels) {
  List<int> best = [0, 0, 0];
  List<int> current = [0, 0, 0];
  for (int i = 0; i + 2 < pixels.length; i += 3) {
    int r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
    int dominant = (r >= g && r >= b) ? 0 : (g >= b ? 1 : 2);
    int spread = (r - g).abs() + (g - b).abs() + (b - r).abs();
    for (int c = 0; c < 3; c++) {
      if (c == dominant) {
        current[c] = spread <= 30 ? current[c] + 2 : current[c] + 1;
      } else if (spread < 10) {
        current[c] = current[c] > 0 ? current[c] - 1 : 0;
      } else {
        current[c] = 0;
      }
      if (current[c] > best[c]) best[c] = current[c];
    }
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(dominantRgbPersistence([]).toString() == '[0, 0, 0]');
  assert(dominantRgbPersistence([5, 1, 1]).toString() == '[2, 0, 0]');
  assert(dominantRgbPersistence([1, 5, 5]).toString() == '[0, 2, 0]');
  print('All tests passed!');
}