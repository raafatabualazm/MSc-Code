@pragma('vm:entry-point')
int countPasswordViolations(String password) {
  int whitespace = 0;
  bool hasUpper = false, hasDigit = false;
  for (int i = 0; i < password.length; i++) {
    int c = password.codeUnitAt(i);
    if (c == 32 || c == 9 || c == 10 || c == 13) {
      whitespace++;
    } else if (c >= 65 && c <= 90) {
      hasUpper = true;
    } else if (c >= 48 && c <= 57) {
      hasDigit = true;
    }
  }
  int violations = whitespace;
  if (!hasUpper) violations++;
  if (!hasDigit) violations++;
  return violations;
}

@pragma('vm:entry-point')
void main() {
  assert(countPasswordViolations('') == 2);
  assert(countPasswordViolations('A1') == 0);
  assert(countPasswordViolations('a b') == 3);
  print('All tests passed!');
}