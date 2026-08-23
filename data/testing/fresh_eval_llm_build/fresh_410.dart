@pragma('vm:entry-point')
bool canRelayElevatorRequests(List<int> requests, int maxGap) {
  List<bool> stable = List<bool>.filled(requests.length + 1, false);
  stable[0] = true;
  for (int i = 0; i < requests.length; i++) {
    stable[i + 1] = stable[i] &&
        (i == 0 || (requests[i] - requests[i - 1]).abs() <= maxGap);
  }
  return stable[requests.length];
}

@pragma('vm:entry-point')
void main() {
  assert(canRelayElevatorRequests([], 0) == true);
  assert(canRelayElevatorRequests([3, 5, 7], 2) == true);
  assert(canRelayElevatorRequests([1, 4], 2) == false);
  print('All tests passed!');
}