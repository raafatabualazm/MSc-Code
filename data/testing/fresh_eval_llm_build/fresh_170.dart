@pragma('vm:entry-point')
List<int> runLengthEncodePasswordCharTypes(String password) {
  if (password.isEmpty) return [];
  List<int> res = []; int cur = -1, cnt = 0;
  for (int i = 0; i < password.length; i++) {
    int t = (password[i].codeUnitAt(0) >= 65 && password[i].codeUnitAt(0) <= 90) ? 1 :
            (password[i].codeUnitAt(0) >= 97 && password[i].codeUnitAt(0) <= 122) ? 0 :
            (password[i].codeUnitAt(0) >= 48 && password[i].codeUnitAt(0) <= 57) ? 2 : 3;
    if (t == cur) cnt++;
    else { if (cur >= 0) { res.add(cnt); res.add(cur); } cur = t; cnt = 1; }
  }
  res.add(cnt); res.add(cur);
  return res;
}

@pragma('vm:entry-point')
void main() {
  assert(runLengthEncodePasswordCharTypes('a').toString() == '[1, 0]');
  assert(runLengthEncodePasswordCharTypes('Aa').toString() == '[1, 1, 1, 0]');
  assert(runLengthEncodePasswordCharTypes('!a').toString() == '[1, 3, 1, 0]');
  print('All tests passed!');
}