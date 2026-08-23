@pragma('vm:entry-point')
String repeatedPacketSizes(List<int> packetSizes) {
  var counts = <int, int>{};
  for (var size in packetSizes) {
    counts[size] = (counts[size] ?? 0) + 1;
  }
  var repeated = counts.entries
      .where((e) => e.value > 1)
      .toList()
      ..sort((a, b) => a.key.compareTo(b.key));
  return repeated.map((e) => '${e.key}:${e.value}').join(',');
}

@pragma('vm:entry-point')
void main() {
  assert(repeatedPacketSizes([]) == '');
  assert(repeatedPacketSizes([3, 3]) == '3:2');
  assert(repeatedPacketSizes([1, 2, 1, 2, 2]) == '1:2,2:3');
  print('All tests passed!');
}