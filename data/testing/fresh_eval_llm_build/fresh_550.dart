@pragma('vm:entry-point')
String passwordRunsAndTypesCheck(String password, int maxRunLength) {
  List<String> types = ['L', 'U', 'D'];
  for (String t in types) {
    int count = 0;
    int maxRun = 0;
    int currentRun = 0;
    for (int i = 0; i < password.length; i++) {
      int code = password.codeUnitAt(i);
      bool match;
      if (t == 'L') {
        match = code >= 97 && code <= 122;
      } else if (t == 'U') {
        match = code >= 65 && code <= 90;
      } else {
        match = code >= 48 && code <= 57;
      }
      if (match) {
        count++;
        currentRun++;
        if (currentRun > maxRun) maxRun = currentRun;
      } else {
        currentRun = 0;
      }
    }
    if (count == 0) {
      return 'INVALID:MISSING_$t';
    }
    if (maxRun > maxRunLength) {
      return 'INVALID:RUN_${t}_$maxRun';
    }
  }
  return 'VALID';
}

@pragma('vm:entry-point')
void main() {
  assert(passwordRunsAndTypesCheck("aA1", 1) == "VALID");
  assert(passwordRunsAndTypesCheck("", 0) == "INVALID:MISSING_L");
  assert(passwordRunsAndTypesCheck("aa", 1) == "INVALID:RUN_L_2");
  print('All tests passed!');
}