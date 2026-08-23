@pragma('vm:entry-point')
bool validateShippingPattern(String manifest, String pattern) {
  int pi = 0;
  for (int i = 0; i < manifest.length && pi < pattern.length; i++) {
    int c = manifest.codeUnitAt(i);
    if (c >= 65 && c <= 90) {
      if (c == pattern.codeUnitAt(pi)) {
        pi++;
      }
    }
  }
  return pi == pattern.length;
}

@pragma('vm:entry-point')
void main() {
  assert(validateShippingPattern('A12B34C', 'ABC') == true);
  assert(validateShippingPattern('A12B34C', '') == true);
  assert(validateShippingPattern('', 'A') == false);
  print('All tests passed!');
}