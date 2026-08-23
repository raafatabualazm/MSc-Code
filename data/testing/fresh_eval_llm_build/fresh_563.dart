@pragma('vm:entry-point')
bool hasValidRuntimeWithinThreshold(String logLine, double maxRuntime) {
  if (logLine.isEmpty) return false;
  var parts = logLine.split(' ');
  if (parts.length < 3) return false;
  var ts = parts[0], usr = parts[1];
  if (ts.length != 8) return false;
  for (int i = 0; i < 8; i++) { if ((i == 2 || i == 5) ? ts[i] != ':' : (ts.codeUnitAt(i) < 48 || ts.codeUnitAt(i) > 57)) return false; }
  for (int i = 0; i < usr.length; i++) { int c = usr.codeUnitAt(i); if (!((c > 47 && c < 58) || (c > 64 && c < 91) || (c > 96 && c < 123))) return false; }
  int runtimeCount = 0;
  double runtimeValue = 0.0;
  for (int i = 2; i < parts.length; i++) {
    int eq = parts[i].indexOf('=');
    if (eq == -1) return false;
    var key = parts[i].substring(0, eq), val = parts[i].substring(eq + 1);
    if (key.isEmpty || val.isEmpty) return false;
    for (int j = 0; j < key.length; j++) { int c = key.codeUnitAt(j); if (!((c > 47 && c < 58) || (c > 64 && c < 91) || (c > 96 && c < 123))) return false; }
    if (key == 'runtime') {
      if (!val.endsWith('ms')) return false;
      var numStr = val.substring(0, val.length - 2);
      bool valid = true, hasDigit = false, hasDot = false;
      for (int k = 0; k < numStr.length; k++) {
        int c = numStr.codeUnitAt(k);
        if (c > 47 && c < 58) hasDigit = true;
        else if (c == 46) { if (hasDot) { valid = false; break; } hasDot = true; }
        else { valid = false; break; }
      }
      if (!valid || !hasDigit) return false;
      runtimeValue = double.parse(numStr);
      runtimeCount++;
      if (runtimeCount > 1) return false;
    }
  }
  return runtimeCount == 1 && runtimeValue <= maxRuntime;
}

@pragma('vm:entry-point')
void main() {
  assert(hasValidRuntimeWithinThreshold("12:30:45 user1 runtime=5.0ms", 10.0) == true);
  assert(hasValidRuntimeWithinThreshold("12:30:45 user1 runtime=15.0ms", 10.0) == false);
  assert(hasValidRuntimeWithinThreshold("", 0.0) == false);
  print('All tests passed!');
}