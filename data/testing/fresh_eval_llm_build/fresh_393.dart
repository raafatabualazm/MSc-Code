@pragma('vm:entry-point')
int decodeElectionMarginFromPairs(int tally) {
  int lead = 0;
  for (int i = 0; i < 32; i += 2) {
    int pair = (tally >> i) & 3;
    if (pair == 1) {
      lead++;
    } else if (pair == 2) {
      lead--;
    }
  }
  return lead;
}

@pragma('vm:entry-point')
void main() {
  assert(decodeElectionMarginFromPairs(0) == 0);
  assert(decodeElectionMarginFromPairs(0x55555555) == 16);
  assert(decodeElectionMarginFromPairs(0xAAAAAAAA) == -16);
  print('All tests passed!');
}