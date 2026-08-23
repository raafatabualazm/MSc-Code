@pragma('vm:entry-point')
bool hasIndexedGrayPixel(List<int> pixels) {
  int low = 0;
  int high = pixels.length - 1;
  while (low <= high) {
    int mid = (low + high) >> 1;
    int pixel = pixels[mid];
    int r = (pixel >> 16) & 255;
    int g = (pixel >> 8) & 255;
    int b = pixel & 255;
    if (r == g && g == b) {
      if (r == mid) return true;
      if (r < mid) {
        high = mid - 1;
      } else {
        low = mid + 1;
      }
    } else if (((r + g + b) ~/ 3) < mid) {
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(hasIndexedGrayPixel([]) == false);
  assert(hasIndexedGrayPixel([0x000000]) == true);
  assert(hasIndexedGrayPixel([0x000100, 0x030303, 0x050505]) == false);
  print('All tests passed!');
}