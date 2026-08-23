@pragma('vm:entry-point')
bool hasCentralWifiCluster(int bins) {
  int mask = bins & 0xFF;
  int count = 0;
  for (int x = mask; x != 0; x &= x - 1) {
    count++;
  }
  int middle = mask & (mask << 1) & (mask >> 1);
  return count == 3 && middle != 0 && (mask & 0x81) == 0;
}

@pragma('vm:entry-point')
void main() {
  assert(hasCentralWifiCluster(28) == true);
  assert(hasCentralWifiCluster(7) == false);
  assert(hasCentralWifiCluster(42) == false);
  print('All tests passed!');
}