@pragma('vm:entry-point')
List<int> deriveBarcodeFlagTrail(String code) {
  if (code.isEmpty) {
    return [];
  }
  List<int> out = [];
  int flags = 0;
  for (int i = 0; i < code.length; i++) {
    int c = code.codeUnitAt(i) - 48;
    if (c < 0 || c > 9) {
      continue;
    }
    int bit = 1 << c;
    if ((flags & bit) != 0) {
      flags &= ~bit;
    } else {
      flags |= bit;
    }
    int pop = 0;
    for (int b = 0; b < 10; b++) {
      if (((flags >> b) & 1) == 1) {
        pop++;
      }
    }
    int echo = 0;
    for (int j = i - 1; j >= 0 && j >= i - 3; j--) {
      int other = code.codeUnitAt(j) - 48;
      if (other == c) {
        echo++;
      } else if (((other ^ c) & 1) == 0) {
        echo += 2;
      }
    }
    if (pop == 0) {
      out.add(0);
    } else if ((echo & 1) == 1) {
      out.add(((flags << 1) | (flags >> 9)) & 1023);
    } else {
      out.add(flags ^ (1 << (pop - 1)));
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(deriveBarcodeFlagTrail('').toString() == '[]');
  assert(deriveBarcodeFlagTrail('01').toString() == '[0, 1]');
  assert(deriveBarcodeFlagTrail('202').toString() == '[5, 7, 2]');
  print('All tests passed!');
}