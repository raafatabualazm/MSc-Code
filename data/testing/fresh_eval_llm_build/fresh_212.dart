@pragma('vm:entry-point')
List<int> parseWifiBinCounts(String readings) {
  // bins: [excellent(>= -50), good(-51..-68), fair(-69..-80), poor(< -80)]
  // Wait, let me use clean thresholds:
  // excellent: v >= -50
  // good: v >= -68 && v < -50
  // fair: v >= -80 && v < -68
  // poor: v < -80
  List<int> bins = [0, 0, 0, 0];
  int n = readings.length;
  if (n == 0) return bins;
  int i = 0;
  while (i < n) {
    // skip whitespace
    while (i < n && readings[i] == ' ') i++;
    if (i >= n) break;
    // parse token
    bool negative = false;
    if (readings[i] == '-') { negative = true; i++; }
    if (i >= n || (readings[i].codeUnitAt(0) < 48 || readings[i].codeUnitAt(0) > 57)) {
      // skip invalid
      while (i < n && readings[i] != ' ') i++;
      continue;
    }
    int val = 0;
    while (i < n && readings[i].codeUnitAt(0) >= 48 && readings[i].codeUnitAt(0) <= 57) {
      val = val * 10 + (readings[i].codeUnitAt(0) - 48);
      i++;
    }
    int dbm = negative ? -val : val;
    if (dbm >= -50) {
      bins[0]++;
    } else if (dbm >= -68) {
      bins[1]++;
    } else if (dbm >= -80) {
      bins[2]++;
    } else {
      bins[3]++;
    }
  }
  return bins;
}

@pragma('vm:entry-point')
void main() {
  assert(parseWifiBinCounts('').toString() == '[0, 0, 0, 0]');
  assert(parseWifiBinCounts('-45 -80 -30 -95 -65').toString() == '[2, 1, 1, 1]');
  assert(parseWifiBinCounts('-50 -51 -68 -69 -80 -81').toString() == '[1, 2, 2, 1]');
  print('All tests passed!');
}