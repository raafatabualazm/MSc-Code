@pragma('vm:entry-point')
int findFirstBarcodeDigitAboveMidpoint(List<int> barcodeDigits) {
  if (barcodeDigits.isEmpty) return -1;
  int threshold = (barcodeDigits[0] + barcodeDigits[barcodeDigits.length - 1]) ~/ 2;
  int low = 0, high = barcodeDigits.length;
  while (low < high) {
    int mid = (low + high) ~/ 2;
    if (barcodeDigits[mid] > threshold) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return low == barcodeDigits.length ? -1 : low;
}

@pragma('vm:entry-point')
void main() {
  assert(findFirstBarcodeDigitAboveMidpoint([1, 2, 3, 4]) == 2);
  assert(findFirstBarcodeDigitAboveMidpoint([5]) == -1);
  assert(findFirstBarcodeDigitAboveMidpoint([0, 9, 9]) == 1);
  print('All tests passed!');
}