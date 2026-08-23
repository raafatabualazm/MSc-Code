@pragma('vm:entry-point')
int flippedMorseDashCount(int symbol) {
  int masked = symbol & 0x1F;
  int flipped = (~masked) & 0x1F;
  int count = 0;
  while (flipped != 0) {
    count += flipped & 1;
    flipped >>= 1;
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(flippedMorseDashCount(0) == 5);
  assert(flippedMorseDashCount(31) == 0);
  assert(flippedMorseDashCount(21) == 2);
  print('All tests passed!');
}