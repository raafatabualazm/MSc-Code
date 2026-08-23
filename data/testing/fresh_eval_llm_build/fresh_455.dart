@pragma('vm:entry-point')
int rgbDiamondContactScore(List<String> pixels, int radius) {
  int score = 0;
  for (int i = 0; i < pixels.length; i++) {
    var a = pixels[i].split(',');
    int ax = int.parse(a[0]), ay = int.parse(a[1]);
    int ar = int.parse(a[2]), ag = int.parse(a[3]), ab = int.parse(a[4]);
    bool aWarm = ar > ag && ar > ab;
    bool aGreen = ag > ar && ag > ab;
    for (int j = i + 1; j < pixels.length; j++) {
      var b = pixels[j].split(',');
      int d = (ax - int.parse(b[0])).abs() + (ay - int.parse(b[1])).abs();
      bool bWarm = int.parse(b[2]) > int.parse(b[3]) && int.parse(b[2]) > int.parse(b[4]);
      bool bGreen = int.parse(b[3]) > int.parse(b[2]) && int.parse(b[3]) > int.parse(b[4]);
      if (d <= radius) {
        score += aWarm == bWarm ? (aWarm ? radius - d + 1 : 1) : -1;
      } else if (d == radius + 1 && (aGreen || bGreen)) {
        score += 2;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(rgbDiamondContactScore([], 2) == 0);
  assert(rgbDiamondContactScore(['0,0,9,1,0','1,0,8,2,1'], 2) == 2);
  assert(rgbDiamondContactScore(['0,0,1,9,0','2,0,9,1,0'], 1) == 2);
  print('All tests passed!');
}