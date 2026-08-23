@pragma('vm:entry-point')
List<int> adjustThermostatNightFlags(List<int> schedules, int comfortMask) {
  List<int> out = [];
  for (final v in schedules) {
    int overlap = v & comfortMask;
    int count = 0;
    for (int t = overlap; t != 0; t &= t - 1) {
      count++;
    }
    int next;
    if (count >= 3) {
      next = ((v << 1) | (v >> 7)) & 255;
      if ((next & 1) == 1) next ^= comfortMask;
    } else if (count == 0) {
      next = ((v >> 1) | ((v & 1) << 7)) & 255;
      next |= comfortMask & 15;
    } else {
      next = v ^ overlap;
      if ((comfortMask & 128) != 0) next &= 127;
    }
    out.add(next);
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(adjustThermostatNightFlags([0], 5).toString() == '[5]');
  assert(adjustThermostatNightFlags([255], 15).toString() == '[240]');
  assert(adjustThermostatNightFlags([1, 2, 3], 1).length == 3);
  print('All tests passed!');
}