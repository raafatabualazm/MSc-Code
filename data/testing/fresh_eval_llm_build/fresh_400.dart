@pragma('vm:entry-point')
List<String> inspectShiftedPasswordRules(String encodedPassword) {
  bool hasDigit = false;
  bool hasPair = false;
  String decoded = '';
  for (int i = 0; i < encodedPassword.length; i++) {
    int code = encodedPassword.codeUnitAt(i) - 1;
    decoded += String.fromCharCode(code);
    if (code >= 48 && code <= 57) hasDigit = true;
    if (i > 0 && decoded.codeUnitAt(i) == decoded.codeUnitAt(i - 1)) hasPair = true;
  }
  return [if (!hasDigit) 'digit', if (!hasPair) 'pair'];
}

@pragma('vm:entry-point')
void main() {
  assert(inspectShiftedPasswordRules('').toString() == '[digit, pair]');
  assert(inspectShiftedPasswordRules('bb2').toString() == '[]');
  assert(inspectShiftedPasswordRules('bc2').toString() == '[pair]');
  print('All tests passed!');
}