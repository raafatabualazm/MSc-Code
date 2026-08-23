@pragma('vm:entry-point')
bool matchesBarcodeParityBands(String code) {
  if (code.isEmpty) return false;
  int mask = 0;
  int mismatches = 0;
  for (int i = 0; i < code.length; i++) {
    int d = code.codeUnitAt(i) - 48;
    if (d < 0 || d > 9) return false;
    mask ^= 1 << d;
    int bits = 0;
    int temp = mask;
    while (temp != 0) {
      bits++;
      temp &= temp - 1;
    }
    if (((bits ^ d) & 1) != 0) mismatches++;
    if (mismatches > 1) return false;
  }
  return mask != 0 && (mask & (mask - 1)) != 0;
}

@pragma('vm:entry-point')
void main() {
  assert(matchesBarcodeParityBands('12') == true);
  assert(matchesBarcodeParityBands('1245') == false);
  assert(matchesBarcodeParityBands('7') == false);
  print('All tests passed!');
}