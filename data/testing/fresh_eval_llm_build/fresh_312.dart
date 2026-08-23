@pragma('vm:entry-point')
bool matchesTieredPasswordRule(String password) {
  List<String> parts = password.split('#');
  if (parts.length != 3) return false;
  int strongParts = 0;
  for (String part in parts) {
    if (part.isEmpty || part.length < 2) return false;
    bool hasLower = false;
    bool hasDigit = false;
    for (int i = 0; i < part.length; i++) {
      int c = part.codeUnitAt(i);
      if (c >= 97 && c <= 122) {
        hasLower = true;
      } else if (c >= 48 && c <= 57 && i > 0) {
        hasDigit = true;
      } else {
        return false;
      }
      if (i > 0 && part[i] == part[i - 1]) return false;
    }
    if (hasLower && hasDigit) strongParts++;
  }
  return strongParts == 2;
}

@pragma('vm:entry-point')
void main() {
  assert(matchesTieredPasswordRule('ab1#c2#de') == true);
  assert(matchesTieredPasswordRule('a1#b2#c3') == false);
  assert(matchesTieredPasswordRule('1a#bc#d4') == false);
  print('All tests passed!');
}