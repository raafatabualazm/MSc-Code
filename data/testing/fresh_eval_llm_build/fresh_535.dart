@pragma('vm:entry-point')
bool areQrAlignmentPositionsValid(int size, int spacing) {
  if (size <= 0 || spacing <= 0) return false;
  bool isPrime(int n) {
    if (n < 2) return false;
    for (int i = 2; i * i <= n; i++) {
      if (n % i == 0) return false;
    }
    return true;
  }
  for (int i = 1; i < size; i++) {
    for (int j = 1; j < size; j++) {
      if (i % spacing != 0) continue;
      if (j % spacing != 0) continue;
      if (i == 1 || j == 1) return false;
      bool iPrime = isPrime(i);
      bool jPrime = isPrime(j);
      if (iPrime != jPrime) return false;
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(areQrAlignmentPositionsValid(5, 2) == false);
  assert(areQrAlignmentPositionsValid(10, 5) == true);
  assert(areQrAlignmentPositionsValid(-1, 2) == false);
  print('All tests passed!');
}