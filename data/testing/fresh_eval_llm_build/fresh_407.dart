@pragma('vm:entry-point')
int manifestDockResidue(int manifestCode) {
  int n = manifestCode.abs();
  if (n == 0) {
    return 4;
  }
  int seal = 0;
  while (n > 0) {
    int digit = n % 10;
    bool prime = digit > 1;
    for (int d = 2; d * d <= digit; d++) {
      if (digit % d == 0) {
        prime = false;
        break;
      }
    }
    if (prime) {
      seal += digit * 2;
    } else if (digit % 2 == 0) {
      seal += digit + 1;
    } else {
      seal += 3;
    }
    n ~/= 10;
  }
  int residue = 0;
  while (seal > 0) {
    int part = seal % 7;
    residue += part >= 4 ? part - 1 : part + 2;
    seal ~/= 7;
  }
  return residue;
}

@pragma('vm:entry-point')
void main() {
  assert(manifestDockResidue(0) == 4);
  assert(manifestDockResidue(27) == 7);
  assert(manifestDockResidue(-530) == 9);
  print('All tests passed!');
}