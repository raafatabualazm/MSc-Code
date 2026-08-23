@pragma('vm:entry-point')
import 'dart:math';

bool isShelfCodeMergeZone(List<int> shelfRects, int requiredArea) {
  if (shelfRects.length != 8) return false;
  int overlapW = max(0, min(shelfRects[2], shelfRects[6]) - max(shelfRects[0], shelfRects[4]));
  int overlapH = max(0, min(shelfRects[3], shelfRects[7]) - max(shelfRects[1], shelfRects[5]));
  int area = overlapW * overlapH;
  int cornerGap = (shelfRects[0] - shelfRects[4]).abs() + (shelfRects[1] - shelfRects[5]).abs();
  return area == requiredArea && cornerGap <= requiredArea;
}

@pragma('vm:entry-point')
void main() {
  assert(isShelfCodeMergeZone([0, 0, 4, 4, 2, 1, 5, 3], 4) == true);
  assert(isShelfCodeMergeZone([0, 0, 4, 4, 2, 1, 5, 3], 3) == false);
  assert(isShelfCodeMergeZone([0, 0, 0, 3, 0, 0, 3, 0], 0) == true);
  print('All tests passed!');
}