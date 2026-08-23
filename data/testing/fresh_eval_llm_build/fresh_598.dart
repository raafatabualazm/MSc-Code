@pragma('vm:entry-point')
int qrModuleDayDrift(String encodedRows, int resetGap) {
  if (resetGap < 0) return -1;
  if (encodedRows.isEmpty) return 0;
  int score = 0;
  List<String> rows = encodedRows.split('|');
  for (String row in rows) {
    if (row.isEmpty) continue;
    List<String> parts = row.split(',');
    List<int> days = [];
    for (String p in parts) {
      int v = int.parse(p);
      if (v < 0) return -1;
      days.add(v);
    }
    for (int i = 0; i < days.length; i++) {
      for (int j = 0; j < i; j++) {
        int gap = days[i] - days[j];
        if (gap < 0) return -1;
        if (gap == 0) continue;
        if (gap <= resetGap) {
          score += gap * (i - j);
        } else if ((gap - resetGap).isEven) {
          score -= (gap - resetGap) ~/ 2;
        } else {
          score -= gap - resetGap;
        }
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(qrModuleDayDrift("", 3) == 0);
  assert(qrModuleDayDrift("0,2|1,4", 2) == 1);
  assert(qrModuleDayDrift("2,1", 5) == -1);
  print('All tests passed!');
}