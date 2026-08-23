@pragma('vm:entry-point')
bool hasComplementarySensorParity(int packet) {
  int left = packet & 0xFFFF;
  int right = (packet >> 16) & 0xFFFF;
  int popcount(int x) {
    int count = 0;
    while (x != 0) {
      x &= x - 1;
      count++;
    }
    return count;
  }
  return (popcount(left) & 1) != (popcount(right) & 1);
}

@pragma('vm:entry-point')
void main() {
  assert(hasComplementarySensorParity(0) == false);
  assert(hasComplementarySensorParity(0x0001FFFF) == true);
  assert(hasComplementarySensorParity(0x80000000) == true);
  print('All tests passed!');
}