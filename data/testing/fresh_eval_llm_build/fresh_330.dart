@pragma('vm:entry-point')
double wifiZoneArea(List<int> points) {
  if (points.length < 6 || points.length % 2 != 0) return 0.0;
  int n = points.length ~/ 2;
  int sum = 0;
  for (int i = 0; i < n; i++) {
    int x = points[2 * i];
    int y = points[2 * i + 1];
    int next = (i + 1) % n;
    int nx = points[2 * next];
    int ny = points[2 * next + 1];
    sum += x * ny - nx * y;
  }
  return (sum.abs() / 2.0);
}

@pragma('vm:entry-point')
void main() {
  assert(wifiZoneArea([]) == 0.0);
  assert(wifiZoneArea([0, 0, 1, 0, 0, 1]) == 0.5);
  assert(wifiZoneArea([1, 1, 2, 2, 3, 3]) == 0.0);
  print('All tests passed!');
}