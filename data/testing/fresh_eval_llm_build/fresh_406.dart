@pragma('vm:entry-point')
int countBarcodeDigitsMatchingValue(String barcode) {
  var counts = <String, int>{};
  var matches = 0;
  for (var ch in barcode.split('')) {
    var next = (counts[ch] ?? 0) + 1;
    counts[ch] = next;
    var digit = ch.codeUnitAt(0) - 48;
    if (digit > 0 && next == digit) matches++;
    if (digit > 0 && next == digit + 1) matches--;
  }
  return matches;
}

@pragma('vm:entry-point')
void main() {
  assert(countBarcodeDigitsMatchingValue('1') == 1);
  assert(countBarcodeDigitsMatchingValue('222') == 0);
  assert(countBarcodeDigitsMatchingValue('3334444') == 2);
  print('All tests passed!');
}