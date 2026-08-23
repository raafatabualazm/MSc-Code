@pragma('vm:entry-point')
int longestMoistureWindowInTolerance(List<int> gridReadings) {
  if (gridReadings.length < 2) return 0;
  final int tolerance = gridReadings.last;
  final List<int> readings = gridReadings.sublist(0, gridReadings.length - 1);
  int left = 0;
  int best = 0;
  for (int right = 0; right < readings.length; right++) {
    int lo = readings[right], hi = readings[right];
    for (int k = left; k <= right; k++) {
      if (readings[k] < lo) lo = readings[k];
      if (readings[k] > hi) hi = readings[k];
    }
    while (hi - lo > tolerance) {
      left++;
      lo = readings[left]; hi = readings[left];
      for (int k = left; k <= right; k++) {
        if (readings[k] < lo) lo = readings[k];
        if (readings[k] > hi) hi = readings[k];
      }
    }
    if (right - left + 1 > best) best = right - left + 1;
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(longestMoistureWindowInTolerance([3, 1, 4, 1, 5, 2]) == 2);
  assert(longestMoistureWindowInTolerance([5, 5, 5, 0]) == 3);
  assert(longestMoistureWindowInTolerance([1, 2, 3, 4, 5, 2]) == 3);
  print('All tests passed!');
}