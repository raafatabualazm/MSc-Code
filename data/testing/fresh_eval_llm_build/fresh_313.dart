@pragma('vm:entry-point')
String warehouseCycleFlags(String bins, int mask) {
  int active = 0;
  int delayed = 0;
  int rotated = mask & 255;
  for (int i = 0; i < bins.length; i++) {
    int lane = bins.codeUnitAt(i) & 7;
    int bit = 1 << lane;
    if ((rotated & bit) != 0) {
      active++;
      rotated = ((rotated << 1) | (rotated >> 7)) & 255;
      delayed += (bins.codeUnitAt(i) & 1) == 0 ? lane : -1;
    } else {
      delayed += (lane & 1) == 0 ? 2 : 1;
      rotated ^= bit;
    }
  }
  int setBits = 0;
  for (int v = rotated; v != 0; v >>= 1) {
    if ((v & 1) != 0) setBits++;
  }
  return '$active|$delayed|$setBits';
}

@pragma('vm:entry-point')
void main() {
  assert(warehouseCycleFlags('', 0) == '0|0|0');
  assert(warehouseCycleFlags('AB', 2) == '2|1|1');
  assert(warehouseCycleFlags('HH', 1) == '1|2|2');
  print('All tests passed!');
}