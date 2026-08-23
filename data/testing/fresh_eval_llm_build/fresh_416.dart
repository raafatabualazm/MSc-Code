@pragma('vm:entry-point')
double averageSeatChartActiveDays(List<int> chartDays) {
  if (chartDays.length < 2) {
    return 0.0;
  }
  int totalDays = 0;
  int intervalCount = 0;
  for (int i = 0; i + 1 < chartDays.length; i += 2) {
    totalDays += (chartDays[i + 1] - chartDays[i]).abs();
    intervalCount++;
  }
  return totalDays / intervalCount;
}

@pragma('vm:entry-point')
void main() {
  assert(averageSeatChartActiveDays([2, 5]) == 3.0);
  assert(averageSeatChartActiveDays([0, 0, 2, 3]) == 0.5);
  assert(averageSeatChartActiveDays([10, 14, 20, 21, 30, 34, 99]) == 3.0);
  print('All tests passed!');
}