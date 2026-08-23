@pragma('vm:entry-point')
num findPacketThresholdSize(List<int> packetSizes) {
  if (packetSizes.isEmpty) return -1;
  int total = 0;
  int minSize = packetSizes[0];
  int maxSize = packetSizes[0];
  for (int s in packetSizes) {
    total += s;
    if (s < minSize) minSize = s;
    if (s > maxSize) maxSize = s;
  }
  if (total == 0) return 0;
  int target = (total + 1) ~/ 2;
  int low = minSize;
  int high = maxSize;
  int result = maxSize;
  while (low <= high) {
    int mid = low + (high - low) ~/ 2;
    int sum = 0;
    for (int s in packetSizes) {
      if (s <= mid) {
        sum += s;
      }
    }
    if (sum >= target) {
      result = mid;
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(findPacketThresholdSize([5, 10, 15]) == 10);
  assert(findPacketThresholdSize([1, 2, 3, 100]) == 100);
  assert(findPacketThresholdSize([]) == -1);
  print('All tests passed!');
}