@pragma('vm:entry-point')
bool hasNonContiguousSetpoints(List<int> schedule) {
  Map<int, List<int>> tempToHours = {};
  for (int i = 0; i < schedule.length; i++) {
    tempToHours.putIfAbsent(schedule[i], () => []).add(i);
  }
  for (var hours in tempToHours.values) {
    if (hours.length > 1) {
      for (int j = 1; j < hours.length; j++) {
        if (hours[j] != hours[j-1] + 1) {
          return true;
        }
      }
    }
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(hasNonContiguousSetpoints([]) == false);
  assert(hasNonContiguousSetpoints([70, 72, 70]) == true);
  assert(hasNonContiguousSetpoints([70, 70, 72, 72]) == false);
  print('All tests passed!');
}