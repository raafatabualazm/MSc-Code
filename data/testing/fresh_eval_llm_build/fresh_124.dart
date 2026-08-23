@pragma('vm:entry-point')
int longestCarryoverWifiSpan(List<int> bins, int minimumSignal) {
  int best = 0;
  int span = 0;
  for (final signal in bins) {
    if (signal >= minimumSignal) {
      span++;
    } else if (signal == minimumSignal - 1 && span > 0) {
      span++;
    } else {
      span = 0;
    }
    if (span > best) best = span;
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(longestCarryoverWifiSpan([], 5) == 0);
  assert(longestCarryoverWifiSpan([5, 4, 5], 5) == 3);
  assert(longestCarryoverWifiSpan([4, 5, 3, 5], 5) == 1);
  print('All tests passed!');
}