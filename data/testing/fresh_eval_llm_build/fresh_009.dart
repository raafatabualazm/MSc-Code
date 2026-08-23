@pragma('vm:entry-point')
int roundCentsByPopcountParity(int cents, int unit) {
  int pop = 0, n = cents;
  while (n != 0) {
    pop += n & 1;
    n >>= 1;
  }
  int remainder = cents & (unit - 1);
  if (remainder == 0) return cents;
  return pop.isOdd ? cents + unit - remainder : cents - remainder;
}

@pragma('vm:entry-point')
void main() {
  assert(roundCentsByPopcountParity(5, 4) == 4);
  assert(roundCentsByPopcountParity(7, 4) == 8);
  assert(roundCentsByPopcountParity(0, 4) == 0);
  print('All tests passed!');
}