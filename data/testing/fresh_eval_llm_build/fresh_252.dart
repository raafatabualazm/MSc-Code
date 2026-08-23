@pragma('vm:entry-point')
String locateBalancedPacketBurst(List<int> packetSizes) {
  int left = 0;
  int total = 0;
  for (int right = 0; right < packetSizes.length; right++) {
    total += packetSizes[right];
    if (total == 1024) return '$left:$right';
    if (total > 1024 && left <= right) total -= packetSizes[left++];
  }
  return 'clear';
}

@pragma('vm:entry-point')
void main() {
  assert(locateBalancedPacketBurst([512, 512]) == '0:1');
  assert(locateBalancedPacketBurst([800, 300, 224]) == 'clear');
  assert(locateBalancedPacketBurst([600, 500, 524]) == '1:2');
  print('All tests passed!');
}