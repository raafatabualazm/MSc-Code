@pragma('vm:entry-point')
List<int> passwordRuleDefects(List<String> passwords) {
  int scan(String s, int l, int r) {
    if (l >= r) return 0;
    if (r - l == 1) {
      var c = s.codeUnitAt(l);
      if (c >= 65 && c <= 90) return 1;
      if (c >= 97 && c <= 122) return 2;
      if (c >= 48 && c <= 57) return 4;
      return 8;
    }
    var m = (l + r) >> 1;
    return scan(s, l, m) | scan(s, m, r);
  }

  var out = <int>[];
  for (var s in passwords) {
    var mask = scan(s, 0, s.length), bad = 0;
    if ((mask & 1) == 0) bad++;
    if ((mask & 2) == 0) bad++;
    if ((mask & 4) == 0) bad++;
    for (var i = 0; i + 2 < s.length; i++) {
      if (s[i] == s[i + 1] && s[i + 1] == s[i + 2]) {
        bad += 2;
      } else if (i + 3 < s.length && s[i] == s[i + 3]) {
        bad++;
      }
    }
    out.add(bad);
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(passwordRuleDefects(['Aa1']).toString() == '[0]');
  assert(passwordRuleDefects(['aaa']).toString() == '[4]');
  assert(passwordRuleDefects(['abca']).toString() == '[3]');
  print('All tests passed!');
}