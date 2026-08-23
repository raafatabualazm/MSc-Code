@pragma('vm:entry-point')
int longestSteadyThermostatSpan(List<int> targets) {
  int left = 0, best = 0;
  for (int right = 0; right < targets.length; right++) {
    if (right > 0 && (targets[right] - targets[right - 1]).abs() > 2) {
      left = right;
    }
    best = best > right - left + 1 ? best : right - left + 1;
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(longestSteadyThermostatSpan([]) == 0);
  assert(longestSteadyThermostatSpan([20, 22, 24]) == 3);
  assert(longestSteadyThermostatSpan([20, 23]) == 1);
  print('All tests passed!');
}