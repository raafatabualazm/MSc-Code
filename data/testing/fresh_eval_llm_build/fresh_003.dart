@pragma('vm:entry-point')
int countDeparturesInMinuteRange(String timetable, int lowerMin, int upperMin) {
  if (timetable.isEmpty) return 0;
  int count = 0;
  for (final entry in timetable.split(',')) {
    final parts = entry.trim().split(':');
    if (parts.length < 2) continue;
    final minutes = int.parse(parts[1]);
    if (minutes >= lowerMin && minutes <= upperMin) count++;
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countDeparturesInMinuteRange('08:15,09:30,10:05,14:45', 10, 30) == 2);
  assert(countDeparturesInMinuteRange('', 0, 59) == 0);
  assert(countDeparturesInMinuteRange('06:30,07:30,08:30', 30, 30) == 3);
  print('All tests passed!');
}