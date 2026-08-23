@pragma('vm:entry-point')
int spellEditDriftMask(String edits) {
  int mask = 0;
  int penalty = 0;
  for (int i = 0; i < edits.length; i++) {
    int c = edits.codeUnitAt(i) | 32;
    if (c >= 97 && c <= 122) {
      int bit = 1 << ((c - 97) & 15);
      bool vowel = c == 97 || c == 101 || c == 105 || c == 111 || c == 117;
      if (vowel) {
        mask = ((mask << 1) | (mask >> 15)) & 0xFFFF;
        if ((mask & bit) != 0) {
          penalty += 2;
        } else {
          mask ^= bit;
        }
      } else if ((mask & bit) == 0) {
        mask |= bit;
      } else {
        mask &= ~bit;
        penalty++;
      }
    } else if (c == 42) {
      mask ^= 0xAAAA;
    }
  }
  int pop = 0;
  while (mask != 0) {
    pop += mask & 1;
    mask >>= 1;
  }
  return pop + penalty;
}

@pragma('vm:entry-point')
void main() {
  assert(spellEditDriftMask('') == 0);
  assert(spellEditDriftMask('bb') == 1);
  assert(spellEditDriftMask('*a') == 10);
  print('All tests passed!');
}