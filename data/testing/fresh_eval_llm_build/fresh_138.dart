@pragma('vm:entry-point')
String platformMinuteTrendCode(List<List<int>> timetableMinutes) {
  var code = '';
  for (final row in timetableMinutes) {
    if (row.isEmpty) {
      code += '-';
    } else {
      code += row.last > row.first ? 'L' : row.last < row.first ? 'E' : 'S';
    }
  }
  return code;
}

@pragma('vm:entry-point')
void main() {
  assert(platformMinuteTrendCode([]) == '');
  assert(platformMinuteTrendCode([[12, 15], [30, 30], [9, 4]]) == 'LSE');
  assert(platformMinuteTrendCode([[], [5], [8, 2, 8]]) == '-SS');
  print('All tests passed!');
}