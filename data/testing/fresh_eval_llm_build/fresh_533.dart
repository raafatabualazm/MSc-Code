@pragma('vm:entry-point')
List<int> categorizePasswordAuditResults(List<String> passwords) {
  List<int> totals = [0, 0, 0, 0, 0];
  for (String password in passwords) {
    if (password.isEmpty) {
      totals[1]++;
      continue;
    }
    bool shortRule = password.length < 6, pairRule = false, seqRule = false;
    bool hasUpper = false, hasLower = false, hasDigit = false;
    for (int i = 0; i < password.length; i++) {
      int c = password.codeUnitAt(i);
      if (c >= 65 && c <= 90) hasUpper = true;
      if (c >= 97 && c <= 122) hasLower = true;
      if (c >= 48 && c <= 57) hasDigit = true;
      for (int j = i + 1; j < password.length && j <= i + 2; j++) {
        if (password[i] == password[j]) pairRule = true;
      }
      if (i + 2 < password.length) {
        int a = password.codeUnitAt(i), b = password.codeUnitAt(i + 1), d = password.codeUnitAt(i + 2);
        if (b == a + 1 && d == b + 1) seqRule = true;
      }
    }
    bool typeRule = !(hasUpper && hasLower && hasDigit);
    if (shortRule) totals[1]++;
    if (pairRule) totals[2]++;
    if (seqRule) totals[3]++;
    if (typeRule) totals[4]++;
    if (!shortRule && !pairRule && !seqRule && !typeRule) totals[0]++;
  }
  return totals;
}

@pragma('vm:entry-point')
void main() {
  assert(categorizePasswordAuditResults(['Aa1b2C']).toString() == '[1, 0, 0, 0, 0]');
  assert(categorizePasswordAuditResults(['']).toString() == '[0, 1, 0, 0, 0]');
  assert(categorizePasswordAuditResults(['aaB345', 'ABCD12']).toString() == '[0, 0, 1, 2, 1]');
  print('All tests passed!');
}