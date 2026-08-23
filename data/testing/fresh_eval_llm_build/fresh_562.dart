@pragma('vm:entry-point')
String packetIntervalLedger(List<int> sizes) {
  if (sizes.isEmpty) return 'none';
  int stable = 0;
  int unstable = 0;
  int special = 0;
  int longest = 0;
  for (int start = 0; start < sizes.length; start++) {
    int sum = 0;
    for (int end = start; end < sizes.length; end++) {
      int value = sizes[end];
      if (value < 0) break;
      sum += value;
      int span = end - start + 1;
      if (sum == 0) continue;
      if (sum < 1024 && span > longest) {
        longest = span;
      }
      if (sum % span == 0) {
        int avg = sum ~/ span;
        if (avg >= 128 && avg <= 256) {
          stable++;
        } else if (avg > 512) {
          unstable += span;
        } else {
          unstable++;
        }
      } else if (span > 2 && sum % 64 == 0) {
        special++;
      }
    }
  }
  return '$stable:$unstable:$special:$longest';
}

@pragma('vm:entry-point')
void main() {
  assert(packetIntervalLedger([128]) == '1:0:0:1');
  assert(packetIntervalLedger([1, 1, 62]) == '0:4:1:3');
  assert(packetIntervalLedger([128, -1, 128]) == '2:0:0:1');
  print('All tests passed!');
}