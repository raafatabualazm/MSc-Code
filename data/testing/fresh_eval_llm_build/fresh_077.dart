@pragma('vm:entry-point')
List<double> tideCrossingPair(List<double> readings) {
  int low = 0;
  int high = readings.length;
  while (low < high) {
    int mid = (low + high) >> 1;
    if (readings[mid] < 0.0) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  if (low == 0 || low == readings.length) return [];
  return [readings[low - 1], readings[low]];
}

@pragma('vm:entry-point')
void main() {
  assert(tideCrossingPair([-2.0, -0.5, 0.0, 1.0]).toString() == '[-0.5, 0.0]');
  assert(tideCrossingPair([-1.0, -0.5]).toString() == '[]');
  assert(tideCrossingPair([]).length == 0);
  print('All tests passed!');
}