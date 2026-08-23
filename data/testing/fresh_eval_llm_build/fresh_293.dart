@pragma('vm:entry-point')
import 'dart:math';

List<int> thermostatScheduleOverlapStats(List<int> dayZone, List<int> nightZone) {
  int left = max(dayZone[0], nightZone[0]), right = min(dayZone[2], nightZone[2]);
  int bottom = max(dayZone[1], nightZone[1]), top = min(dayZone[3], nightZone[3]);
  if (right > left && top > bottom) {
    return [(right - left) * (top - bottom), 0];
  }
  int dx = max(0, max(nightZone[0] - dayZone[2], dayZone[0] - nightZone[2]));
  int dy = max(0, max(nightZone[1] - dayZone[3], dayZone[1] - nightZone[3]));
  return [0, dx + dy];
}

@pragma('vm:entry-point')
void main() {
  assert(thermostatScheduleOverlapStats([0, 0, 4, 4], [2, 1, 5, 3]).toString() == '[4, 0]');
  assert(thermostatScheduleOverlapStats([0, 0, 2, 2], [5, 0, 7, 2]).toString() == '[0, 3]');
  assert(thermostatScheduleOverlapStats([0, 0, 1, 1], [1, 1, 3, 3]).toString() == '[0, 0]');
  print('All tests passed!');
}