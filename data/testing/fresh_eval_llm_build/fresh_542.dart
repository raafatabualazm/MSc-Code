@pragma('vm:entry-point')
bool isValidTimetableMinutes(List<List<int>> timetable) {
  if (timetable.isEmpty) return true;
  int M = timetable.length;
  int N = timetable[0].length;
  for (int i = 0; i < M; i++) {
    if (timetable[i].length != N) return false;
    for (int j = 0; j < N; j++) {
      int t = timetable[i][j];
      if (t < 0 || t > 1440) return false;
    }
  }
  for (int i = 0; i < M - 1; i++) {
    int minDiff = 1441;
    int maxDiff = 0;
    for (int j = 0; j < N; j++) {
      int a = timetable[i][j];
      int b = timetable[i + 1][j];
      if (a >= b) return false;
      int diff = b - a;
      if (diff < minDiff) minDiff = diff;
      if (diff > maxDiff) maxDiff = diff;
    }
    if (maxDiff - minDiff > 15) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isValidTimetableMinutes([]) == true);
  assert(isValidTimetableMinutes([[600, 610], [620, 630]]) == true);
  assert(isValidTimetableMinutes([[600, 610], [600, 620]]) == false);
  print('All tests passed!');
}