@pragma('vm:entry-point')
int qrTimingStripeResidue(int modules) {
  int n = modules.abs();
  int residue = 1;
  while (n > 0) {
    int digit = n % 5;
    if (digit != 0) {
      residue = (residue * digit) % 7;
    }
    n ~/= 5;
  }
  return residue;
}

@pragma('vm:entry-point')
void main() {
  assert(qrTimingStripeResidue(0) == 1);
  assert(qrTimingStripeResidue(24) == 2);
  assert(qrTimingStripeResidue(-38) == 6);
  print('All tests passed!');
}