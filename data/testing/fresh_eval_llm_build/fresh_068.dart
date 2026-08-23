@pragma('vm:entry-point')
bool hasRecoverableThermostatWindow(List<int> schedule) {
  int left = 0;
  int sum = 0;
  bool sawHeat = false;
  bool sawCool = false;
  for (int right = 0; right < schedule.length; right++) {
    int value = schedule[right];
    sum += value;
    sawHeat = sawHeat || value > 0;
    sawCool = sawCool || value < 0;
    while (sum > 2 || sum < -2) {
      int removed = schedule[left++];
      sum -= removed;
      sawHeat = schedule.sublist(left, right + 1).any((v) => v > 0);
      sawCool = schedule.sublist(left, right + 1).any((v) => v < 0);
    }
    if (sum == 0 && sawHeat && sawCool && right - left >= 2) return true;
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(hasRecoverableThermostatWindow([1, 1, -2]) == true);
  assert(hasRecoverableThermostatWindow([2, 1, -3]) == false);
  assert(hasRecoverableThermostatWindow([0, 0, 0]) == false);
  print('All tests passed!');
}