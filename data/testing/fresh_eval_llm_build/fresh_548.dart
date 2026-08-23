@pragma('vm:entry-point')
List<num> auditNickelPrimeRounding(List<int> cents) {
  int netShift = 0;
  int primeUps = 0;
  int gcdValue = 0;
  for (final value in cents) {
    if (value == 0) continue;
    int sign = value < 0 ? -1 : 1;
    int absValue = value.abs();
    int rem = absValue % 5;
    int rounded = absValue;
    if (rem == 1) {
      rounded--;
    } else if (rem == 4) {
      rounded++;
    } else if (rem == 2 || rem == 3) {
      bool prime = absValue > 1;
      for (int d = 2; d * d <= absValue; d++) {
        if (absValue % d == 0) {
          prime = false;
          break;
        }
      }
      if ((rem == 2 && prime) || (rem == 3 && !prime)) {
        rounded += 5 - rem;
        if (rem == 2 && prime) primeUps++;
      } else {
        rounded -= rem;
      }
    }
    rounded *= sign;
    netShift += rounded - value;
    int g = rounded.abs();
    if (g == 0) continue;
    while (g != 0 && gcdValue != 0) {
      int t = gcdValue % g;
      gcdValue = g;
      g = t;
    }
    if (gcdValue == 0) gcdValue = rounded.abs();
  }
  return [netShift, primeUps, gcdValue];
}

@pragma('vm:entry-point')
void main() {
  assert(auditNickelPrimeRounding([]).toString() == '[0, 0, 0]');
  assert(auditNickelPrimeRounding([2, 7, 17]).toString() == '[9, 3, 5]');
  assert(auditNickelPrimeRounding([6, 9]).toString() == '[0, 0, 5]');
  print('All tests passed!');
}