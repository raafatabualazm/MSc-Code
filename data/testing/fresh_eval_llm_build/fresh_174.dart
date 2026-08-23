@pragma('vm:entry-point')
String reviewPasswordCipher(String password, int shift) {
  var out = StringBuffer();
  int lower = 0, upper = 0, digit = 0, special = 0;
  for (int i = 0; i < password.length; i++) {
    int c = password.codeUnitAt(i);
    if (c >= 97 && c <= 122) {
      out.writeCharCode(97 + (c - 97 + shift) % 26);
      lower++;
    } else if (c >= 65 && c <= 90) {
      out.writeCharCode(65 + (c - 65 + shift) % 26);
      upper++;
    } else if (c >= 48 && c <= 57) {
      out.write(i > 0 && password[i] == password[i - 1] ? '*' : String.fromCharCode(48 + (c - 48 + shift) % 10));
      digit++;
    } else {
      out.write('_');
      special++;
    }
  }
  int groups = (lower > 0 ? 1 : 0) + (upper > 0 ? 1 : 0) + (digit > 0 ? 1 : 0) + (special > 0 ? 1 : 0);
  String level = password.length >= 8 && groups == 4 ? 'STRONG' : (password.length >= 6 && groups >= 2 ? 'MEDIUM' : 'WEAK');
  return '${out.toString()}|$level';
}

@pragma('vm:entry-point')
void main() {
  assert(reviewPasswordCipher('', 3) == '|WEAK');
  assert(reviewPasswordCipher('aA1!', 1) == 'bB2_|WEAK');
  assert(reviewPasswordCipher('Secure77#', 4) == 'Wigyvi1*_|STRONG');
  print('All tests passed!');
}