@pragma('vm:entry-point')
List<int> deriveTelemetryHealthCodes(List<int> samples) {
  List<int> out = [];
  for (int s in samples) {
    int low = s & 15;
    int rotated = ((s << 1) & 255) | ((s >> 7) & 1);
    int bits = 0;
    int t = rotated;
    while (t != 0) {
      bits += t & 1;
      t >>= 1;
    }
    int code;
    if (bits >= 5) {
      code = low ^ bits;
      if ((s & 128) != 0) code += 16;
    } else {
      code = rotated & 31;
      if ((low & 3) == 0) code ^= 8;
    }
    out.add(code);
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(deriveTelemetryHealthCodes([]).toString() == '[]');
  assert(deriveTelemetryHealthCodes([0, 1, 4]).toString() == '[8, 2, 0]');
  assert(deriveTelemetryHealthCodes([255]).toString() == '[23]');
  print('All tests passed!');
}