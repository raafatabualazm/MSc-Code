@pragma('vm:entry-point')
int shelfCodeAlignmentResidue(String shelfCode, int modulus) {
  int modBase = modulus == 0 ? 1 : modulus.abs();
  int score = 0;
  for (int i = 0; i < shelfCode.length; i++) {
    int v = shelfCode.codeUnitAt(i);
    v = (v >= 48 && v <= 57) ? v - 48 : (v & 31) + 9;
    bool prime = v > 1;
    for (int d = 2; d * d <= v; d++) {
      if (v % d == 0) {
        prime = false;
        break;
      }
    }
    if (prime) {
      score += (v * (i + 1)) % modBase;
    } else if (v % 2 == 0) {
      score -= v ~/ 2;
    } else {
      score += v % modBase;
    }
  }
  int a = score.abs(), b = modulus.abs();
  while (b != 0) {
    int t = a % b;
    a = b;
    b = t;
  }
  return a;
}

@pragma('vm:entry-point')
void main() {
  assert(shelfCodeAlignmentResidue("A1", 6) == 2);
  assert(shelfCodeAlignmentResidue("C3", 6) == 6);
  assert(shelfCodeAlignmentResidue("H2", 6) == 3);
  print('All tests passed!');
}