@pragma('vm:entry-point')
int barcodeWeightedGcd(List<int> digits) {
  int n = digits.length;
  if (n == 0) return 0;
  int weightedSum = 0;
  for (int i = 0; i < n; i++) {
    weightedSum += digits[i] * (i % 2 == 0 ? 1 : 3);
  }
  int a = weightedSum, b = n;
  while (b != 0) {
    int t = b;
    b = a % b;
    a = t;
  }
  return a;
}

@pragma('vm:entry-point')
void main() {
  assert(barcodeWeightedGcd([]) == 0);
  assert(barcodeWeightedGcd([2,4,6,8]) == 4);
  assert(barcodeWeightedGcd([0,0,0]) == 3);
  print('All tests passed!');
}