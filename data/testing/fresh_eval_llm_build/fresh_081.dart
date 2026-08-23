@pragma('vm:entry-point')
String? longestWeakSignalRunLabel(List<int> rssiReadings) {
  int bestStart = -1, bestLen = 0, runStart = -1, runLen = 0;
  for (int i = 0; i < rssiReadings.length; i++) {
    if (rssiReadings[i] >= -80 && rssiReadings[i] <= -67) {
      if (runLen == 0) runStart = i;
      runLen++;
      if (runLen > bestLen) { bestLen = runLen; bestStart = runStart; }
    } else {
      runLen = 0;
    }
  }
  if (bestLen == 0) return null;
  return 'days $bestStart-${bestStart + bestLen - 1}: weak';
}

@pragma('vm:entry-point')
void main() {
  assert(longestWeakSignalRunLabel([]) == null);
  assert(longestWeakSignalRunLabel([-70]) == 'days 0-0: weak');
  assert(longestWeakSignalRunLabel([-50, -70, -75, -80, -67, -60]) == 'days 1-4: weak');
  print('All tests passed!');
}