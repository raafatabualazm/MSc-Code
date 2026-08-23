@pragma('vm:entry-point')
String annotatePackedTelemetry(List<int> samples) {
  var buf = StringBuffer();
  for (var s in samples) {
    bool crit = (s & 0x80000000) != 0;
    bool warn = (s & 0x40000000) != 0;
    int v = s & 0xFF;
    int pop = 0;
    while (v != 0) {
      pop++;
      v &= v - 1;
    }
    if (crit) {
      buf.write(pop.isOdd ? 'C' : 'c');
    } else if (warn) {
      buf.write(pop > 4 ? 'W' : 'w');
    } else {
      buf.write(pop % 10);
    }
  }
  return buf.toString();
}

@pragma('vm:entry-point')
void main() {
  assert(annotatePackedTelemetry([]) == "");
  assert(annotatePackedTelemetry([0x80000001, 0x4000001F]) == "CW");
  assert(annotatePackedTelemetry([0]) == "0");
  print('All tests passed!');
}