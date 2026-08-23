@pragma('vm:entry-point')
bool areElevatorFloorsBase3Palindromic(List<int> floors) {
  for (int floor in floors) {
    if (floor < 0) return false;
    int n = floor;
    if (n == 0) continue;
    var digits = <int>[];
    while (n > 0) {
      digits.add(n % 3);
      n ~/= 3;
    }
    for (int i = 0; i < digits.length ~/ 2; i++) {
      if (digits[i] != digits[digits.length - 1 - i]) {
        return false;
      }
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(areElevatorFloorsBase3Palindromic([0, 1, 2, 4, 8]) == true);
  assert(areElevatorFloorsBase3Palindromic([3, 4]) == false);
  assert(areElevatorFloorsBase3Palindromic([]) == true);
  print('All tests passed!');
}