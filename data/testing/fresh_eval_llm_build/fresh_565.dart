@pragma('vm:entry-point')
List<int> morseSymbolFlagTrail(String tape) {
  List<int> out = [];
  if (tape.isEmpty) return out;
  for (String part in tape.split(' ')) {
    if (part.isEmpty) continue;
    int mask = 1;
    int dots = 0;
    for (int i = 0; i < part.length; i++) {
      String ch = part[i];
      if (ch == '.') {
        mask = ((mask << 1) ^ 1) & 255;
        dots++;
      } else if (ch == '-') {
        mask = ((mask << 2) | 2) & 255;
      } else if (ch == '/') {
        mask = ((mask >> 1) | 128) & 255;
      } else {
        continue;
      }
      int bits = 0;
      for (int t = mask; t != 0; t >>= 1) {
        bits += t & 1;
      }
      if (bits.isEven) {
        mask = ((mask << 1) | (mask >> 7)) & 255;
      } else if (dots > 2 && (mask & 1) == 1) {
        mask ^= 85;
      }
    }
    if ((mask & 15) == 0) continue;
    out.add(part.length.isOdd ? (mask ^ part.length) : (mask + dots));
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(morseSymbolFlagTrail('.').toString() == '[7]');
  assert(morseSymbolFlagTrail('/').length == 0);
  assert(morseSymbolFlagTrail('.- ..').toString() == '[27, 15]');
  print('All tests passed!');
}