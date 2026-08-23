@pragma('vm:entry-point')
String orderPacketsByMedianGap(List<int> packetSizes) {
  var sorted = List<int>.from(packetSizes);
  sorted.sort((a, b) {
    var da = (a - 512).abs();
    var db = (b - 512).abs();
    return da == db ? b.compareTo(a) : da.compareTo(db);
  });
  return sorted.isEmpty ? 'empty' : sorted.join(':');
}

@pragma('vm:entry-point')
void main() {
  assert(orderPacketsByMedianGap([]) == 'empty');
  assert(orderPacketsByMedianGap([511, 513]) == '513:511');
  assert(orderPacketsByMedianGap([520, 500, 512]) == '512:520:500');
  print('All tests passed!');
}