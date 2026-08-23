@pragma('vm:entry-point')
List<String> auditPasswordTokens(String source) {
  List<String> out = [];
  for (String entry in source.split(';')) {
    if (entry.isEmpty) continue;
    int pivot = -1;
    for (int i = 0; i < entry.length; i++) {
      if (entry[i] == '=') {
        if (pivot != -1) {
          pivot = -2;
          break;
        }
        pivot = i;
      }
    }
    if (pivot <= 0 || pivot >= entry.length - 1) {
      out.add('syntax');
      continue;
    }
    String user = entry.substring(0, pivot);
    String pass = entry.substring(pivot + 1);
    bool lower = false, upper = false, digit = false, bad = false;
    for (int i = 0; i < user.length; i++) {
      int u = user.codeUnitAt(i);
      if (u < 97 || u > 122) bad = true;
    }
    int run = 1;
    for (int i = 0; i < pass.length && !bad; i++) {
      String c = pass[i];
      if (i > 0) run = pass[i - 1] == c ? run + 1 : 1;
      if (run >= 3 || c == ' ') {
        bad = true;
        break;
      }
      if (c.compareTo('a') >= 0 && c.compareTo('z') <= 0) lower = true;
      else if (c.compareTo('A') >= 0 && c.compareTo('Z') <= 0) upper = true;
      else if (c.compareTo('0') >= 0 && c.compareTo('9') <= 0) digit = true;
      else {
        for (String mark in ['!', '@', '#']) {
          if (c == mark && i + 1 < pass.length && pass[i + 1] == mark) bad = true;
        }
      }
    }
    out.add(!bad && pass.length >= 4 && pass.length <= 8 && lower && upper && digit ? '$user:OK' : '$user:weak');
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(auditPasswordTokens('sam=Ab3d').toString() == '[sam:OK]');
  assert(auditPasswordTokens('x=AA11').toString() == '[x:weak]');
  assert(auditPasswordTokens('bad==Aa1b').toString() == '[syntax]');
  print('All tests passed!');
}