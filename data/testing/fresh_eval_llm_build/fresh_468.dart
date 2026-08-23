@pragma('vm:entry-point')
bool thermostatScheduleBuffersValid(List<List<int>> weekday, List<List<int>> weekend) {
  for (final block in weekday) {
    if (block.length != 4 || block[0] > block[1]) {
      return false;
    }
  }
  for (final block in weekend) {
    if (block.length != 4 || block[0] > block[1]) {
      return false;
    }
  }
  for (final a in weekday) {
    for (final b in weekend) {
      int distance = (a[2] - b[2]).abs() + (a[3] - b[3]).abs();
      int latestStart = a[0] > b[0] ? a[0] : b[0];
      int earliestEnd = a[1] < b[1] ? a[1] : b[1];
      if (latestStart < earliestEnd) {
        if (distance <= 2) {
          return false;
        }
        continue;
      }
      if (latestStart == earliestEnd) {
        if (distance == 0) {
          return false;
        }
        if (distance == 1 && (a[0] + a[1]) % 2 == (b[0] + b[1]) % 2) {
          return false;
        }
        continue;
      }
      int gap = latestStart - earliestEnd;
      if (gap == 1 && distance < 2) {
        return false;
      }
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(thermostatScheduleBuffersValid([], []) == true);
  assert(thermostatScheduleBuffersValid([[0, 5, 0, 0]], [[3, 6, 2, 0]]) == false);
  assert(thermostatScheduleBuffersValid([[0, 2, 0, 0]], [[3, 5, 2, 0]]) == true);
  print('All tests passed!');
}