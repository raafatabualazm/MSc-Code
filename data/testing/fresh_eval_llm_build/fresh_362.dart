@pragma('vm:entry-point')
List<String> asciiIntervalRows(int startDay, int dayCount) {
  List<String> rows = [];
  for (int i = 0; i < dayCount; i++) {
    int day = startDay + i;
    String edge = day % 7 == 0 ? '+' : '|';
    rows.add('$edge${List.filled(day.abs() % 4 + 1, '-').join()}$day');
  }
  return rows;
}

@pragma('vm:entry-point')
void main() {
  assert(asciiIntervalRows(0, 0).toString() == '[]');
  assert(asciiIntervalRows(6, 3).toString() == '[|---6, +----7, |-8]');
  assert(asciiIntervalRows(-2, 4).toString() == '[|----2, |---1, +-0, |--1]');
  print('All tests passed!');
}