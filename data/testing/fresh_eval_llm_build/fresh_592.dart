@pragma('vm:entry-point')
int countWifiCoverageBins(List<int> data) {
  if (data.length < 1) return 0;
  int t = data[0];
  int n = (data.length - 1) ~/ 3;
  if (n == 0) return 0;
  int minX = data[1], maxX = data[1], minY = data[2], maxY = data[2];
  for (int i = 1; i < data.length; i += 3) {
    int x = data[i], y = data[i + 1], r = data[i + 2];
    if (x - r < minX) minX = x - r;
    if (x + r > maxX) maxX = x + r;
    if (y - r < minY) minY = y - r;
    if (y + r > maxY) maxY = y + r;
  }
  if (t <= 0) return (maxX - minX + 1) * (maxY - minY + 1);
  int total = 0;
  for (int x = minX; x <= maxX; x++) {
    for (int y = minY; y <= maxY; y++) {
      int c = 0;
      for (int i = 1; i < data.length; i += 3) {
        int dx = (x - data[i]).abs(), dy = (y - data[i + 1]).abs();
        if (dx + dy <= data[i + 2]) {
          if (++c >= t) {
            total++;
            break;
          }
        }
      }
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(countWifiCoverageBins([]) == 0);
  assert(countWifiCoverageBins([3]) == 0);
  assert(countWifiCoverageBins([1, 0, 0, 1]) == 5);
  print('All tests passed!');
}