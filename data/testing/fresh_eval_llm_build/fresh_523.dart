@pragma('vm:entry-point')
String classifyServerLogWave(String logs) {
  int ok = 0, warn = 0, severe = 0, bad = 0, streak = 0;
  for (String line in logs.split('\n')) {
    if (line.isEmpty) continue;
    int at = line.indexOf('@'), gt = line.indexOf('>');
    if (at <= 0 || gt <= at + 1 || gt == line.length - 1) {
      bad++;
      streak = 0;
      continue;
    }
    String type = line.substring(0, at);
    int id = 0;
    for (int i = at + 1; i < gt; i++) {
      int c = line.codeUnitAt(i);
      if (c < 48 || c > 57) {
        id = -1;
        break;
      }
      id = id * 10 + c - 48;
    }
    if (id < 0) {
      bad++;
      streak = 0;
      continue;
    }
    bool panic = false, invalid = false;
    String token = '';
    for (int i = gt + 1; i <= line.length; i++) {
      String ch = i == line.length ? ',' : line[i];
      if (ch == ',') {
        if (token == 'panic') panic = true;
        token = '';
      } else if (ch.codeUnitAt(0) < 97 || ch.codeUnitAt(0) > 122) {
        invalid = true;
        break;
      } else {
        token += ch;
      }
    }
    if (invalid) {
      bad++;
      streak = 0;
      continue;
    }
    if (type == 'OK') {
      ok++;
      streak = 0;
    } else if (type == 'WARN') {
      warn++;
      streak = 0;
    } else if (type == 'FAIL') {
      if (id >= 500 || panic) {
        severe++;
        streak++;
      } else {
        streak = 0;
      }
    } else {
      bad++;
      streak = 0;
    }
    if (streak >= 2) return 'ALARM:$ok/$warn/$severe/$bad';
  }
  return 'CLEAR:$ok/$warn/$severe/$bad';
}

@pragma('vm:entry-point')
void main() {
  assert(classifyServerLogWave('') == 'CLEAR:0/0/0/0');
  assert(classifyServerLogWave('FAIL@500>disk\nFAIL@1>panic') == 'ALARM:0/0/2/0');
  assert(classifyServerLogWave('OK@2>Up') == 'CLEAR:0/0/0/1');
  print('All tests passed!');
}