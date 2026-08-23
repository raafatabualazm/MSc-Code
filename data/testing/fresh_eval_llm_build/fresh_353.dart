@pragma('vm:entry-point')
List<int> buildPrimeBaseEncodings(List<List<int>> grid, int base) {
  var result = <int>[];
  for (var row in grid) {
    for (var cell in row) {
      if (cell < 2) continue;
      bool isPrime = true;
      for (var d = 2; d * d <= cell; d++) {
        if (cell % d == 0) {
          isPrime = false;
          break;
        }
      }
      if (!isPrime) continue;
      int n = cell, encoded = 0, mult = 1;
      while (n > 0) {
        int digit = n % base;
        encoded += digit * mult;
        mult *= 10;
        n ~/= base;
      }
      result.add(encoded);
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(buildPrimeBaseEncodings([[2,3,4]], 2).length == 2);
  assert(buildPrimeBaseEncodings([[2]], 10)[0] == 2);
  assert(buildPrimeBaseEncodings([[4,6,8]], 2).isEmpty);
  print('All tests passed!');
}