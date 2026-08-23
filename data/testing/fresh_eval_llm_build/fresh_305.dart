@pragma('vm:entry-point')
String advanceShelfCycleCode(String shelfCode, int elapsedDays) {
  int dayIndex = int.parse(shelfCode.substring(1));
  int shifted = (dayIndex + elapsedDays) % 42;
  if (shifted < 0) {
    shifted += 42;
  }
  String zone = shifted < 21 ? "E" : "L";
  return '${shelfCode[0]}${shifted.toString().padLeft(2, '0')}$zone';
}

@pragma('vm:entry-point')
void main() {
  assert(advanceShelfCycleCode('R00', 0) == 'R00E');
  assert(advanceShelfCycleCode('Q41', 1) == 'Q00E');
  assert(advanceShelfCycleCode('M05', -7) == 'M40L');
  print('All tests passed!');
}