@pragma('vm:entry-point')
bool validateBarcodeDigits(String s) {
  if (s.length != 30) return false;
  if (s.substring(0, 3) != '110') return false;
  if (s.substring(27) != '101') return false;
  int sum = 0;
  for (int i = 0; i < 6; i++) {
    int digit = 0;
    for (int j = 0; j < 4; j++) {
      if (s[3 + i * 4 + j] == '1') {
        digit = digit * 2 + 1;
      } else if (s[3 + i * 4 + j] == '0') {
        digit = digit * 2;
      } else {
        return false;
      }
    }
    if (digit > 9) return false;
    sum += digit;
  }
  return sum % 10 == 0;
}

@pragma('vm:entry-point')
void main() {
  assert(validateBarcodeDigits("110000000000000000000000000101") == true);
  assert(validateBarcodeDigits("") == false);
  assert(validateBarcodeDigits("110000100010001000100010001101") == false);
  print('All tests passed!');
}