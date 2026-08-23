@pragma('vm:entry-point')
int countCompletedChargeCycles(List<int> readings, int peakThreshold) {
  int cycles = 0;
  // States: 0 = waiting for below-threshold start,
  //         1 = below threshold (ready to climb),
  //         2 = at/above threshold (peaked, waiting to fall)
  int state = 0;
  for (int i = 0; i < readings.length; i++) {
    int val = readings[i];
    if (state == 0) {
      if (val < peakThreshold) {
        state = 1;
      }
    } else if (state == 1) {
      if (val >= peakThreshold) {
        state = 2;
      }
    } else {
      if (val < peakThreshold) {
        cycles++;
        state = 1;
      }
    }
  }
  return cycles;
}

@pragma('vm:entry-point')
void main() {
  assert(countCompletedChargeCycles([20, 80, 30, 90, 10], 70) == 2);
  assert(countCompletedChargeCycles([80, 20, 80, 20], 70) == 1);
  assert(countCompletedChargeCycles([], 70) == 0);
  print('All tests passed!');
}