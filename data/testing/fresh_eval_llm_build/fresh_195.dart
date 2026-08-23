@pragma('vm:entry-point')
String squareResidueBeacon(String square) {
  if (square.length != 2) return 'invalid';
  int file = square.codeUnitAt(0) - 96;
  int rank = square.codeUnitAt(1) - 48;
  int value = file * file + rank * rank;
  for (int d = 2; d * d <= value; d++) {
    if (value % d == 0) return '${value.toRadixString(8)}:C';
  }
  return '${value.toRadixString(8)}:${value > 1 ? 'P' : 'U'}';
}

@pragma('vm:entry-point')
void main() {
  assert(squareResidueBeacon('a1') == '2:P');
  assert(squareResidueBeacon('b2') == '10:C');
  assert(squareResidueBeacon('h8') == '200:C');
  print('All tests passed!');
}