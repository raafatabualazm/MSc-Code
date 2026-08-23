@pragma('vm:entry-point')
List<int> nickelPrimeRoundingBands(List<int> cents) {
  bool isPrime(int x) {
    if (x < 2) return false;
    for (int d = 2; d * d <= x; d++) {
      if (x % d == 0) return false;
    }
    return true;
  }

  List<int> out = [];
  for (int value in cents) {
    int sign = value < 0 ? -1 : 1;
    int n = value.abs();
    int rem = n % 5;
    int rounded = rem < 3 ? n - rem : n + (5 - rem);
    if (rounded == 0) {
      out.add(0);
      continue;
    }
    int chosen = rounded;
    if (!isPrime(rounded ~/ 5)) {
      for (int step = 1; step <= 3; step++) {
        int lower = rounded - step * 5;
        int upper = rounded + step * 5;
        bool lowerPrime = lower >= 10 && isPrime(lower ~/ 5);
        bool upperPrime = isPrime(upper ~/ 5);
        if (lowerPrime && upperPrime) {
          chosen = rem == 3 ? upper : lower;
          break;
        } else if (lowerPrime || upperPrime) {
          chosen = lowerPrime ? lower : upper;
          break;
        }
      }
    }
    out.add(chosen * sign);
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(nickelPrimeRoundingBands([20, 43]).toString() == '[15, 55]');
  assert(nickelPrimeRoundingBands([1, 5, 10]).toString() == '[0, 10, 10]');
  assert(nickelPrimeRoundingBands([-21, 45]).toString() == '[-15, 35]');
  print('All tests passed!');
}