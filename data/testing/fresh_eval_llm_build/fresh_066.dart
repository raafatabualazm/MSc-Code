@pragma('vm:entry-point')
int accumulatedSatelliteSilence(List<int> passDays) {
  int delayedDays = 0;
  for (int i = 1; i < passDays.length; i++) {
    int silentGap = passDays[i] - passDays[i - 1] - 1;
    if (silentGap > 2) {
      delayedDays += silentGap - 2;
    }
  }
  return delayedDays;
}

@pragma('vm:entry-point')
void main() {
  assert(accumulatedSatelliteSilence([]) == 0);
  assert(accumulatedSatelliteSilence([1, 5]) == 1);
  assert(accumulatedSatelliteSilence([0, 4, 10]) == 4);
  print('All tests passed!');
}