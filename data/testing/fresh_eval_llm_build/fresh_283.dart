@pragma('vm:entry-point')
List<int> qrModuleResiduePrimes(String modules, int maxPrime) {
  List<int> result = [];
  if (maxPrime < 2) {
    return result;
  }
  for (int p = 2; p <= maxPrime; p++) {
    bool prime = true;
    for (int d = 2; d * d <= p; d++) {
      if (p % d == 0) {
        prime = false;
        break;
      }
    }
    if (!prime) {
      continue;
    }
    int rem = 0;
    int dark = 0;
    for (int i = 0; i < modules.length; i++) {
      int bit;
      if (modules[i] == '#') {
        bit = 1;
        dark++;
      } else if (modules[i] == '.') {
        bit = 0;
      } else {
        continue;
      }
      rem = (rem * 2 + bit) % p;
    }
    if (dark == 0) {
      if (rem == 0) {
        result.add(p);
      }
      continue;
    }
    int target = dark % p;
    if (rem == target || (modules.length % p == 1 && (rem + 1) % p == target)) {
      result.add(p);
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(qrModuleResiduePrimes("", 5).toString() == "[2, 3, 5]");
  assert(qrModuleResiduePrimes("#.", 5).length == 0);
  assert(qrModuleResiduePrimes("#.#", 7).toString() == "[2, 3]");
  print('All tests passed!');
}