@pragma('vm:entry-point')
bool hasPrimeMarkedDeck(List<int> piles) {
  if (piles.isEmpty) return false;
  int g = 0;
  int active = 0;
  for (final pile in piles) {
    if (pile < 0) return false;
    if (pile == 0) continue;
    active++;
    int a = g == 0 ? pile : g;
    int b = g == 0 ? 0 : pile;
    while (b != 0) {
      int t = a % b;
      a = b;
      b = t;
    }
    g = a;
  }
  if (active < 2 || g < 2) return false;
  for (int p = 2; p <= g; p++) {
    if (g % p != 0) continue;
    bool prime = true;
    for (int d = 2; d * d <= p; d++) {
      if (p % d == 0) {
        prime = false;
        break;
      }
    }
    if (!prime) continue;
    if (p >= 13) return false;
    bool seen = false;
    for (final pile in piles) {
      int n = pile;
      while (n > 0) {
        if (n % 13 == p) {
          seen = true;
          break;
        }
        n ~/= 13;
      }
      if (seen) break;
    }
    if (!seen) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(hasPrimeMarkedDeck([30, 42]) == true);
  assert(hasPrimeMarkedDeck([26, 52]) == false);
  assert(hasPrimeMarkedDeck([0, 0]) == false);
  print('All tests passed!');
}