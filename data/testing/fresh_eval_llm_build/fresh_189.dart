@pragma('vm:entry-point')
String selectFirstCompliantPassword(String candidates, String separator, String mandatoryTypes) {
  for (String token in candidates.split(separator)) {
    bool hasDigits = !mandatoryTypes.contains('d') || token.contains(RegExp(r'\d'));
    bool hasUpper = !mandatoryTypes.contains('u') || token.contains(RegExp(r'[A-Z]'));
    bool hasLower = !mandatoryTypes.contains('l') || token.contains(RegExp(r'[a-z]'));
    if (hasDigits && hasUpper && hasLower) {
      return token;
    }
  }
  return "no match";
}

@pragma('vm:entry-point')
void main() {
  assert(selectFirstCompliantPassword("abc|A1|def", "|", "du") == "A1");
  assert(selectFirstCompliantPassword("", ":", "d") == "no match");
  assert(selectFirstCompliantPassword("hello", ",", "") == "hello");
  print('All tests passed!');
}