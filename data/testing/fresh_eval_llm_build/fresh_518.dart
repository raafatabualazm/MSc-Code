@pragma('vm:entry-point')
List<String> filterSafePasswords(List<String> passwords) {
  const forbidden = ['abc', 'admin', 'pass', 'root', 'test', 'user'];
  var safe = <String>[];
  if (passwords.isEmpty) return safe;
  int minForbiddenLen = forbidden.map((s) => s.length).reduce((a,b) => a < b ? a : b);
  for (var pwd in passwords) {
    if (pwd.length < minForbiddenLen) {
      safe.add(pwd);
      continue;
    }
    bool isSafe = true;
    for (int i = 0; i <= pwd.length - minForbiddenLen; i++) {
      for (int len = minForbiddenLen; len <= pwd.length - i; len++) {
        String sub = pwd.substring(i, i + len);
        int low = 0, high = forbidden.length - 1;
        while (low <= high) {
          int mid = (low + high) ~/ 2;
          int cmp = sub.compareTo(forbidden[mid]);
          if (cmp == 0) {
            isSafe = false;
            break;
          } else if (cmp < 0) {
            high = mid - 1;
          } else {
            low = mid + 1;
          }
        }
        if (!isSafe) break;
      }
      if (!isSafe) break;
    }
    if (isSafe) safe.add(pwd);
  }
  return safe;
}

@pragma('vm:entry-point')
void main() {
  assert(filterSafePasswords([]).length == 0);
  assert(filterSafePasswords(['abc']).length == 0);
  assert(filterSafePasswords(['hey']).toString() == '[hey]');
  print('All tests passed!');
}