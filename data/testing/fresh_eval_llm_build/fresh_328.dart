@pragma('vm:entry-point')
String auditPasswordReplay(String script) {
  List<String> pass = [];
  List<String> removed = [];
  List<int> blocks = [];
  for (int i = 0; i < script.length; i++) {
    String ch = script[i];
    if ((ch.codeUnitAt(0) >= 48 && ch.codeUnitAt(0) <= 57) ||
        (ch.codeUnitAt(0) >= 65 && ch.codeUnitAt(0) <= 90) ||
        (ch.codeUnitAt(0) >= 97 && ch.codeUnitAt(0) <= 122)) {
      pass.add(ch);
      continue;
    }
    if (ch == '(') {
      blocks.add(pass.length);
    } else if (ch == ')') {
      if (blocks.isEmpty) return 'SCRIPT_ERROR';
      bool digitSeen = false;
      for (int j = blocks.removeLast(); j < pass.length; j++) {
        int code = pass[j].codeUnitAt(0);
        if (code >= 48 && code <= 57) {
          digitSeen = true;
          break;
        }
      }
      if (!digitSeen) return 'BLOCK_MISSING_DIGIT';
    } else if (ch == '<') {
      if (blocks.isNotEmpty || pass.isEmpty) return 'SCRIPT_ERROR';
      removed.add(pass.removeLast());
    } else if (ch == '^') {
      if (removed.isNotEmpty) pass.add(removed.removeLast());
    } else {
      return 'SCRIPT_ERROR';
    }
  }
  if (blocks.isNotEmpty) return 'SCRIPT_ERROR';
  if (pass.length < 6 || pass.length > 10) return 'WEAK';
  bool lower = false, upper = false, digit = false;
  for (int i = 0; i < pass.length; i++) {
    int code = pass[i].codeUnitAt(0);
    if (code >= 97 && code <= 122) lower = true;
    if (code >= 65 && code <= 90) upper = true;
    if (code >= 48 && code <= 57) digit = true;
    if (i > 1 && pass[i] == pass[i - 1] && pass[i] == pass[i - 2]) return 'WEAK';
  }
  return (lower && upper && digit) ? 'STRONG' : 'WEAK';
}

@pragma('vm:entry-point')
void main() {
  assert(auditPasswordReplay("Ab1cdE") == 'STRONG');
  assert(auditPasswordReplay("ab1") == 'WEAK');
  assert(auditPasswordReplay("A1(xx)") == 'BLOCK_MISSING_DIGIT');
  print('All tests passed!');
}