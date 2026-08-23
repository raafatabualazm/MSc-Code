@pragma('vm:entry-point')
String auditShelfCodeIntervals(List<int> codes) {
  if (codes.isEmpty) return 'none';
  int weekly = 0, overdue = 0, rush = 0, invalid = 0;
  for (int i = 1; i < codes.length; i++) {
    int gap = codes[i] - codes[i - 1];
    if (gap <= 0) {
      invalid++;
    } else if (gap % 7 == 0) {
      if (gap == 7) {
        weekly += 2;
      } else {
        weekly++;
      }
    } else if (gap > 9) {
      overdue += gap ~/ 5;
    } else {
      rush += gap < 3 ? 2 : 1;
    }
  }
  for (int code in codes) {
    if (code < 0) {
      invalid++;
    } else if (code % 30 == 0 && code != 0) {
      overdue++;
    }
  }
  return 'W$weekly-O$overdue-R$rush-I$invalid';
}

@pragma('vm:entry-point')
void main() {
  assert(auditShelfCodeIntervals([]) == 'none');
  assert(auditShelfCodeIntervals([0, 7]) == 'W2-O0-R0-I0');
  assert(auditShelfCodeIntervals([-1, -1, 6]) == 'W2-O0-R0-I3');
  print('All tests passed!');
}