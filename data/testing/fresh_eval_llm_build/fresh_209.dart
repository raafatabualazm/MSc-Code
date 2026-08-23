@pragma('vm:entry-point')
int countSinglePassageCells(int packed) {
  int count = 0;
  for (int i = 0; i < 8; i++) {
    int cell = (packed >> (i * 4)) & 0xF;
    bool nsPair = ((cell & 0x1) == 0) && ((cell & 0x4) == 0);
    bool ewPair = ((cell & 0x2) == 0) && ((cell & 0x8) == 0);
    if (nsPair ^ ewPair) count++;
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countSinglePassageCells(0x00000000) == 0);
  assert(countSinglePassageCells(0x5A) == 2);
  assert(countSinglePassageCells(0x55555555) == 8);
  print('All tests passed!');
}