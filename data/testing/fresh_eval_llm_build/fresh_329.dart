@pragma('vm:entry-point')
bool hasSafeSensorSpacing(int packed) {
  int reading = packed & 0xFF;
  int rotateBy = (packed >> 8) & 0x7;
  int rotated = ((reading << rotateBy) | (reading >> (8 - rotateBy))) & 0xFF;
  return (rotated & (rotated >> 1)) == 0;
}

@pragma('vm:entry-point')
void main() {
  assert(hasSafeSensorSpacing(0) == true);
  assert(hasSafeSensorSpacing(0xFF) == false);
  assert(hasSafeSensorSpacing(0x55 | (1 << 8)) == true);
  print('All tests passed!');
}