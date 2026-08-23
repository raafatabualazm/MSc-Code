@pragma('vm:entry-point')
int countFrequentOddDominantColors(List<int> pixels) {
  Map<int, int> freq = {};
  for (int p in pixels) {
    int v = p & 0xFFFFFF;
    freq[v] = (freq[v] ?? 0) + 1;
  }
  int count = 0;
  for (int p in freq.keys) {
    if ((freq[p] ?? 0) >= 2) {
      int r = (p >> 16) & 0xFF;
      int g = (p >> 8) & 0xFF;
      int b = p & 0xFF;
      int dominant;
      if (r >= g && r >= b) {
        dominant = r;
      } else if (g >= r && g >= b) {
        dominant = g;
      } else {
        dominant = b;
      }
      if (dominant % 2 != 0) {
        count++;
      }
    }
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countFrequentOddDominantColors([]) == 0);
  assert(countFrequentOddDominantColors([0x123456, 0x123456]) == 0);
  assert(countFrequentOddDominantColors([0x123457, 0x123457]) == 1);
  print('All tests passed!');
}