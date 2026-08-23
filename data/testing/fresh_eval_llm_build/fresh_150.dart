@pragma('vm:entry-point')
bool hasRgbDiagonalTag(int pixel) {
  int r = (pixel >> 16) & 0xFF;
  int g = (pixel >> 8) & 0xFF;
  int b = pixel & 0xFF;
  int flagCount = ((r >> 5) & 1) + ((g >> 5) & 1) + ((b >> 5) & 1);
  int mix = (r ^ g ^ b) & 0x0F;
  int ones = 0;
  while (mix != 0) {
    ones += mix & 1;
    mix >>= 1;
  }
  return flagCount == 1 && ones == 3;
}

@pragma('vm:entry-point')
void main() {
  assert(hasRgbDiagonalTag(0x200007) == true);
  assert(hasRgbDiagonalTag(0x200003) == false);
  assert(hasRgbDiagonalTag(-1) == false);
  print('All tests passed!');
}