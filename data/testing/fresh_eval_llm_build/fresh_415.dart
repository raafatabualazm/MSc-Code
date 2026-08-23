@pragma('vm:entry-point')
int barcodeResiduePeak(String digits) {
  int residue = 0;
  int peaks = 0;
  for (int i = 0; i < digits.length; i++) {
    residue = (residue * 3 + digits.codeUnitAt(i) - 48) % 11;
    if (residue == 2 || residue == 3 || residue == 5 || residue == 7) {
      peaks++;
    }
  }
  return peaks;
}

@pragma('vm:entry-point')
void main() {
  assert(barcodeResiduePeak('') == 0);
  assert(barcodeResiduePeak('39') == 2);
  assert(barcodeResiduePeak('7777') == 3);
  print('All tests passed!');
}