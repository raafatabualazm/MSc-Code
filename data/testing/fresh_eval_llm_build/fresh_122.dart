@pragma('vm:entry-point')
bool isElectionResultGcdPrime(List<int> tallies) {
  int gcd(int a, int b) {
    while (b != 0) { int t = b; b = a % b; a = t; }
    return a;
  }
  bool isPrime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i * i <= n; i += 2) {
      if (n % i == 0) return false;
    }
    return true;
  }
  List<int> nonZero = tallies.where((v) => v > 0).toList();
  if (nonZero.isEmpty) return false;
  int g = nonZero[0];
  int total = 0;
  for (int v in tallies) { total += v; }
  for (int v in nonZero) { g = gcd(g, v); }
  if (total % g != 0) return false;
  int quotient = total ~/ g;
  if (!isPrime(quotient)) return false;
  for (int i = 0; i < nonZero.length; i++) {
    bool dominates = true;
    for (int j = 0; j < nonZero.length; j++) {
      if (i == j) continue;
      if (nonZero[j] % nonZero[i] != 0) { dominates = false; break; }
    }
    if (dominates) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isElectionResultGcdPrime([6, 10, 15]) == true);
  assert(isElectionResultGcdPrime([2, 4]) == false);
  assert(isElectionResultGcdPrime([]) == false);
  print('All tests passed!');
}