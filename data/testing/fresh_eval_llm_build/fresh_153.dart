@pragma('vm:entry-point')
bool isConfirmedExclusivePhase(int phaseCode) {
  int lowerNibble = phaseCode & 0xF;
  int upperNibble = (phaseCode >> 4) & 0xF;
  // Count set bits in lower nibble (active phases)
  int activeBits = 0;
  int tmp = lowerNibble;
  while (tmp > 0) { activeBits += tmp & 1; tmp >>= 1; }
  // Count set bits in upper nibble (sensor confirmations)
  int sensorBits = 0;
  tmp = upperNibble;
  while (tmp > 0) { sensorBits += tmp & 1; tmp >>= 1; }
  return activeBits == 1 && sensorBits >= 2;
}

@pragma('vm:entry-point')
void main() {
  assert(isConfirmedExclusivePhase(0x31) == true);
  assert(isConfirmedExclusivePhase(0x11) == false);
  assert(isConfirmedExclusivePhase(0x33) == false);
  print('All tests passed!');
}