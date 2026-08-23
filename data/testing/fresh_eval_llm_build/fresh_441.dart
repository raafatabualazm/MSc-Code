@pragma('vm:entry-point')
String? encodeTrafficPhasePattern(int phaseTicks) {
  if (phaseTicks < 0) return null;
  if (phaseTicks == 0) return 'G';
  const symbols = ['G', 'Y', 'R'];
  var result = '';
  while (phaseTicks > 0) {
    result = symbols[phaseTicks % 3] + result;
    phaseTicks ~/= 3;
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(encodeTrafficPhasePattern(0) == 'G');
  assert(encodeTrafficPhasePattern(5) == 'YR');
  assert(encodeTrafficPhasePattern(-2) == null);
  print('All tests passed!');
}