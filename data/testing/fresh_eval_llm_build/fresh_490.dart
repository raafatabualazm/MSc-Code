@pragma('vm:entry-point')
List<int> reconcileLedgerBitFlags(List<int> entries) {
  List<int> flags = [];
  int rolling = 0;
  for (int value in entries) {
    int amount = value < 0 ? -value : value;
    int ones = 0;
    while (amount > 0) {
      ones += amount & 1;
      amount >>= 1;
    }
    int mask = value < 0 ? 1 : 0;
    if (ones >= 3) {
      mask |= 2;
      if ((((value < 0 ? -value : value)) & 3) == 3) {
        mask ^= 8;
      }
    } else if (ones == 0) {
      mask |= 4;
    } else {
      mask |= 16;
    }
    rolling = ((rolling << 1) ^ mask) & 31;
    flags.add(rolling);
  }
  return flags;
}

@pragma('vm:entry-point')
void main() {
  assert(reconcileLedgerBitFlags([]).toString() == '[]');
  assert(reconcileLedgerBitFlags([0, 1, 7]).toString() == '[4, 24, 26]');
  assert(reconcileLedgerBitFlags([-8, 3]).toString() == '[17, 18]');
  print('All tests passed!');
}