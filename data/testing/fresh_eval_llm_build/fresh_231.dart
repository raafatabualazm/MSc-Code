@pragma('vm:entry-point')
String summarisePacketTransmission(List<int> packetSizes) {
  const int dayTicks = 86400;
  const int fragThreshold = 1472;
  int total = 0;
  int bursts = 0;
  for (final size in packetSizes) {
    total += size;
    if (size > fragThreshold) bursts++;
  }
  final int day = total ~/ dayTicks;
  final int remainder = total % dayTicks;
  return 'day:$day ticks:$remainder bursts:$bursts';
}

@pragma('vm:entry-point')
void main() {
  assert(summarisePacketTransmission([]) == 'day:0 ticks:0 bursts:0');
  assert(summarisePacketTransmission([1472, 1473]) == 'day:0 ticks:2945 bursts:1');
  assert(summarisePacketTransmission([86400, 1]) == 'day:1 ticks:1 bursts:1');
  print('All tests passed!');
}