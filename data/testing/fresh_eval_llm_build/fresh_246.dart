@pragma('vm:entry-point')
List<String> classifyElevatorRingRequests(int requestMask) {
  List<String> out = [];
  int rotated = ((requestMask << 1) | (requestMask >> 7)) & 255;
  for (int floor = 0; floor < 8; floor++) {
    int bit = 1 << floor;
    if ((rotated & bit) != 0) {
      out.add((requestMask & bit) != 0 ? 'hold${floor + 1}' : 'serve${floor + 1}');
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(classifyElevatorRingRequests(0).toString() == '[]');
  assert(classifyElevatorRingRequests(3).toString() == '[hold2, serve3]');
  assert(classifyElevatorRingRequests(255).length == 8);
  print('All tests passed!');
}