@pragma('vm:entry-point')
int computePassWindowScore(String schedule) {
  if (schedule.isEmpty) return 0;
  int total = 0;
  for (String sat in schedule.split(';')) {
    int i = 0, pending = 0, satScore = 0;
    while (i < sat.length) {
      bool isActive = sat[i++] == 'A';
      int start = i;
      while (i < sat.length && sat.codeUnitAt(i) >= 48 && sat.codeUnitAt(i) <= 57) i++;
      int len = int.parse(sat.substring(start, i));
      if (isActive) {
        pending = len * len;
      } else {
        if (pending > 0) {
          if (len > 3) pending ~/= 2;
          satScore += pending;
          pending = 0;
        }
      }
    }
    if (pending > 0) satScore += pending + 10;
    total += satScore;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(computePassWindowScore("") == 0);
  assert(computePassWindowScore("A5") == 35);
  assert(computePassWindowScore("A3I2A4") == 35);
  print('All tests passed!');
}