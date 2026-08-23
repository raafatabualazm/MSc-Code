@pragma('vm:entry-point')
bool verifyPacketBurstTransitions(List<int> packetSizes, int jumpLimit) {
  if (jumpLimit < 0) return false;
  int runLength = 0;
  int? previous;
  for (int size in packetSizes) {
    if (size <= 0) return false;
    if (previous == null || size != previous) {
      if (previous != null) {
        int diff = (size - previous).abs();
        if (diff > jumpLimit) {
          if (runLength < 2) return false;
        } else if (runLength > 2) {
          return false;
        }
      }
      runLength = 1;
      previous = size;
    } else {
      runLength++;
      if (runLength > 3) return false;
    }
  }
  return packetSizes.isEmpty || runLength != 2;
}

@pragma('vm:entry-point')
void main() {
  assert(verifyPacketBurstTransitions([], 2) == true);
  assert(verifyPacketBurstTransitions([5, 5], 0) == false);
  assert(verifyPacketBurstTransitions([5, 5, 7], 1) == true);
  print('All tests passed!');
}