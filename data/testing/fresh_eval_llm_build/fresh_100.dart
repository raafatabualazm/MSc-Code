@pragma('vm:entry-point')
import 'dart:math';

int asciiStrokeBoundingArea(List<String> rows) {
  int minX = 1 << 30, maxX = -1, minY = rows.length, maxY = -1;
  for (int y = 0; y < rows.length; y++) {
    int a = rows[y].indexOf('#');
    if (a >= 0) {
      minX = min(minX, a);
      maxX = max(maxX, rows[y].lastIndexOf('#'));
      minY = min(minY, y);
      maxY = y;
    }
  }
  return maxX < 0 ? 0 : (maxX - minX + 1) * (maxY - minY + 1);
}

@pragma('vm:entry-point')
void main() {
  assert(asciiStrokeBoundingArea([]) == 0);
  assert(asciiStrokeBoundingArea(['#']) == 1);
  assert(asciiStrokeBoundingArea(['..##', '....', '#...']) == 12);
  print('All tests passed!');
}