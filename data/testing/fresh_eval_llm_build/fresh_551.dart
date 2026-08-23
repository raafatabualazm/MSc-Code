@pragma('vm:entry-point')
List<int> roundedCentDiamondAudit(List<int> data) {
  if (data.isEmpty || data.length % 3 != 0) return [0, 0, 0];
  List<int> xs = [], ys = [], rs = [];
  int minX = 1 << 30, minY = 1 << 30;
  int maxX = -(1 << 30), maxY = -(1 << 30);
  for (int i = 0; i < data.length; i += 3) {
    int x = data[i], y = data[i + 1];
    int r = (data[i + 2].abs() + 2) ~/ 5;
    xs.add(x);
    ys.add(y);
    rs.add(r);
    if (x - r < minX) minX = x - r;
    if (y - r < minY) minY = y - r;
    if (x + r > maxX) maxX = x + r;
    if (y + r > maxY) maxY = y + r;
  }
  int covered = 0, overlap = 0, isolated = 0;
  for (int i = 0; i < xs.length; i++) {
    bool touching = false;
    for (int j = 0; j < xs.length; j++) {
      if (i == j) continue;
      int d = (xs[i] - xs[j]).abs() + (ys[i] - ys[j]).abs();
      if (d <= rs[i] + rs[j]) {
        touching = true;
        break;
      }
    }
    if (!touching && rs[i] > 0) isolated++;
  }
  for (int x = minX; x <= maxX; x++) {
    for (int y = minY; y <= maxY; y++) {
      int hits = 0;
      for (int i = 0; i < xs.length; i++) {
        int d = (x - xs[i]).abs() + (y - ys[i]).abs();
        if (d <= rs[i]) hits++;
      }
      if (hits == 0) continue;
      covered++;
      if (hits > 1) overlap++;
    }
  }
  return [covered, overlap, isolated];
}

@pragma('vm:entry-point')
void main() {
  assert(roundedCentDiamondAudit([]).toString() == '[0, 0, 0]');
  assert(roundedCentDiamondAudit([0, 0, 4]).toString() == '[5, 0, 1]');
  assert(roundedCentDiamondAudit([0, 0, 0, 0, 0, 0]).toString() == '[1, 1, 0]');
  print('All tests passed!');
}