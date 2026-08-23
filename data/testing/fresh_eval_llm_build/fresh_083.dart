@pragma('vm:entry-point')
int packChannelMajorityFlags(List<int> pixels) {
  int flags = 0;
  for (int i = 0; i < pixels.length && i < 10; i++) {
    int p = pixels[i];
    for (int c = 0; c < 3; c++) {
      int shift = (2 - c) * 8;
      int channel = (p >> shift) & 0xFF;
      int cnt = 0, v = channel;
      while (v != 0) {
        v &= v - 1;
        cnt++;
      }
      if (cnt > 4) {
        flags |= (1 << (i * 3 + c));
      }
    }
  }
  return flags;
}

@pragma('vm:entry-point')
void main() {
  assert(packChannelMajorityFlags([]) == 0);
  assert(packChannelMajorityFlags([0x0F0F0F]) == 0);
  assert(packChannelMajorityFlags([0xFFFFFF]) == 7);
  print('All tests passed!');
}