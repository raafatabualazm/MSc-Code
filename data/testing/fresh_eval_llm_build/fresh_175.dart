@pragma('vm:entry-point')
bool hasBalancedElevatorRequests(int requests) {
  if (requests < 0) {
    return false;
  }
  int pressed = 0;
  int bounce = 0;
  for (int floor = 0; floor < 16; floor++) {
    int bit = (requests >> floor) & 1;
    if (bit == 1) {
      pressed++;
      bool nextPressed = ((requests >> (floor + 1)) & 1) == 1;
      if (nextPressed && ((requests >> (floor + 2)) & 1) == 1) {
        return false;
      } else if (!nextPressed && ((requests >> (floor + 2)) & 1) == 1) {
        bounce++;
      }
    }
  }
  return pressed >= 2 && pressed <= 6 && bounce <= 1;
}

@pragma('vm:entry-point')
void main() {
  assert(hasBalancedElevatorRequests(3) == true);
  assert(hasBalancedElevatorRequests(7) == false);
  assert(hasBalancedElevatorRequests(819) == true);
  print('All tests passed!');
}