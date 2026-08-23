@pragma('vm:entry-point')
double rgbCascadeBalance(List<List<int>> pixels, int pivot) {
  if (pixels.isEmpty || pivot < 0) return 0.0;
  if (pixels.length == 1) {
    double leaf = 0.0;
    for (int c = 0; c < pixels[0].length && c < 3; c++) {
      int v = pixels[0][c];
      if (v == pivot) {
        leaf += 1.0;
      } else if (v > pivot) {
        leaf += 0.5;
      } else if (pivot - v == 1) {
        leaf += 0.25;
      }
    }
    return leaf;
  }
  int mid = pixels.length ~/ 2;
  double score = 0.0;
  for (int i = 0; i < pixels.length; i++) {
    if (pixels[i].length < 3) continue;
    for (int c = 0; c < 3; c++) {
      int value = pixels[i][c];
      if ((value + c + pivot) % 4 == 0) {
        score += 0.25;
      } else if (value > pivot + 8) {
        score += 0.5;
      } else if (value < pivot - 8) {
        score -= 0.25;
      }
    }
  }
  if (score <= -1.0) return score;
  return score +
      rgbCascadeBalance(pixels.sublist(0, mid), pivot - 1) / 2.0 +
      rgbCascadeBalance(pixels.sublist(mid), pivot - 1) / 2.0;
}

@pragma('vm:entry-point')
void main() {
  assert(rgbCascadeBalance([], 3) == 0.0);
  assert(rgbCascadeBalance([[3, 4, 5]], 4) == 1.75);
  assert(rgbCascadeBalance([[4, 4, 4], [4, 4, 4]], 4) == 2.0);
  print('All tests passed!');
}