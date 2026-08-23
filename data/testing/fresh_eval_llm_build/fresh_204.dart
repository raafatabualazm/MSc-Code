@pragma('vm:entry-point')
bool isTelemetrySamplingValid(List<int> days) {
  if (days.length < 2) return true;
  int firstGap = days[1] - days[0];
  if (firstGap < 1 || firstGap > 30) return false;
  for (int i = 2; i < days.length; i++) {
    if (days[i] - days[i-1] != firstGap) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isTelemetrySamplingValid([1, 2, 3]) == true);
  assert(isTelemetrySamplingValid([1, 8, 15]) == true);
  assert(isTelemetrySamplingValid([1, 2, 4]) == false);
  print('All tests passed!');
}