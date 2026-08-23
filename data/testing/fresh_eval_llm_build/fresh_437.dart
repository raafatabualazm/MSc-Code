@pragma('vm:entry-point')
int countTideFloodWarnings(String tideData) {
  List<int> readings = [];
  int i = 0;
  while (i < tideData.length) {
    int c = tideData.codeUnitAt(i);
    if (c == 32 || c == 44) { i++; continue; }
    bool neg = false;
    if (c == 45) { neg = true; i++; c = i < tideData.length ? tideData.codeUnitAt(i) : 0; }
    if (c >= 48 && c <= 57) {
      int num = 0;
      while (i < tideData.length) {
        int d = tideData.codeUnitAt(i);
        if (d < 48 || d > 57) break;
        num = num * 10 + (d - 48);
        i++;
      }
      readings.add(neg ? -num : num);
    } else {
      while (i < tideData.length) {
        int c2 = tideData.codeUnitAt(i);
        if (c2 == 44 || c2 == 32) break;
        i++;
      }
    }
  }
  int count = 0;
  for (int idx = 0; idx < readings.length; idx++) {
    int j = idx;
    while (j + 1 < readings.length && readings[j] < readings[j + 1]) j++;
    if (j - idx + 1 >= 3 && readings[idx] < 0 && readings[j] > 0) count++;
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countTideFloodWarnings("") == 0);
  assert(countTideFloodWarnings("-1,0,1") == 1);
  print('All tests passed!');
}