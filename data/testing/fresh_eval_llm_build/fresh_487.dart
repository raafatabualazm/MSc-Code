@pragma('vm:entry-point')
String summarizePacketQueueBands(List<int> packetSizes) {
  if (packetSizes.isEmpty) return 'idle';
  List<int> filtered = [];
  for (int size in packetSizes) {
    if (size < 0) continue;
    filtered.add(size);
  }
  if (filtered.isEmpty) return 'blocked';
  filtered.sort((a, b) {
    int ra = a % 5;
    int rb = b % 5;
    if (ra != rb) return rb - ra;
    if (a.isEven != b.isEven) return a.isEven ? -1 : 1;
    return a - b;
  });
  List<String> bands = [];
  int checksum = 0;
  for (int i = 0; i < filtered.length; i++) {
    int streak = 1;
    for (int j = i + 1; j < filtered.length; j++) {
      if (filtered[j] != filtered[i]) break;
      streak++;
    }
    checksum += filtered[i] * streak;
    if (filtered[i] == 0) {
      bands.add('z$streak');
    } else if (filtered[i] < 64) {
      bands.add('s$streak');
    } else if (filtered[i] <= 512) {
      bands.add('m$streak');
    } else {
      bands.add('l$streak');
    }
    if (streak > 1) i += streak - 1;
  }
  if (checksum % 7 == 0) {
    bands.add('sync');
  } else if (checksum % 7 < 3) {
    bands.add('drift');
  }
  return bands.join('|');
}

@pragma('vm:entry-point')
void main() {
  assert(summarizePacketQueueBands([]) == 'idle');
  assert(summarizePacketQueueBands([-4, -1]) == 'blocked');
  assert(summarizePacketQueueBands([0, 5]) == 'z1|s1');
  print('All tests passed!');
}