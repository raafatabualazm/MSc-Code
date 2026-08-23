@pragma('vm:entry-point')
String? locatePacketAveragingSpan(List<int> packets) {
  if (packets.isEmpty) return null;
  for (int i = 0; i < packets.length; i++) {
    if (packets[i] < 0) return null;
    int sum = 0;
    int rising = 0;
    for (int j = i; j < packets.length; j++) {
      int value = packets[j];
      if (value < 0) return null;
      sum += value;
      if (j > i) {
        int diff = value - packets[j - 1];
        if (diff.abs() > 5) break;
        if (diff > 0) rising++;
      }
      int len = j - i + 1;
      if (len < 3 || sum % len != 0) continue;
      bool doubledZero = false;
      for (int k = i + 1; k <= j; k++) {
        if (packets[k] == 0 && packets[k - 1] == 0) {
          doubledZero = true;
          break;
        }
      }
      if (!doubledZero && rising * 2 >= len - 1) return '$i-$j';
    }
  }
  return null;
}

@pragma('vm:entry-point')
void main() {
  assert(locatePacketAveragingSpan([1, 3, 5]) == '0-2');
  assert(locatePacketAveragingSpan([1, 6, 12]) == null);
  assert(locatePacketAveragingSpan([-1, 4, 5]) == null);
  print('All tests passed!');
}