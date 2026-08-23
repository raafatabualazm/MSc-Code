@pragma('vm:entry-point')
bool isElevatorSequenceFeasible(List<int> requests) {
  if (requests.isEmpty) return true;
  final Set<int> seen = {};
  for (int i = 0; i < requests.length; i++) {
    int floor = requests[i];
    if (seen.contains(floor)) {
      return false;
    }
    seen.add(floor);
    if (i > 0) {
      int diff = (floor - requests[i - 1]).abs();
      if (diff > 5) {
        return false;
      }
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isElevatorSequenceFeasible([]) == true);
  assert(isElevatorSequenceFeasible([1, 3, 3, 5]) == false);
  assert(isElevatorSequenceFeasible([1, 4, 9, 14]) == true);
  print('All tests passed!');
}