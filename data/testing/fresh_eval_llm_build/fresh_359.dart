@pragma('vm:entry-point')
int computeTelemetryQuality(List<int> samples) {
  int sum = 0;
  for (int s in samples) {
    if ((s & 0x80000000) != 0) continue;
    if ((s & 0x40000000) != 0) {
      sum += s & 0xFF;
    } else {
      int d = s & 0xFFFF;
      int c = 0, t = d;
      while (t != 0) { c += t & 1; t >>= 1; }
      sum += (c % 2 == 1) ? (d & 0xFF) : -(d & 0xFF);
    }
  }
  return sum;
}

@pragma('vm:entry-point')
void main() {
  assert(computeTelemetryQuality([]) == 0);
  assert(computeTelemetryQuality([0x80000000]) == 0);
  assert(computeTelemetryQuality([0x00000001]) == 1);
  print('All tests passed!');
}