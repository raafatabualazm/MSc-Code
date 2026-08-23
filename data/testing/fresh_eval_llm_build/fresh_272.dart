@pragma('vm:entry-point')
String mirrorSquareTrail(String square, int folds) {
  if (folds <= 0 || square.length != 2) return square;
  var file = String.fromCharCode(104 - (square.codeUnitAt(0) - 97));
  var rank = (9 - int.parse(square.substring(1))).toString();
  return square + '-' + mirrorSquareTrail(file + rank, folds - 1);
}

@pragma('vm:entry-point')
void main() {
  assert(mirrorSquareTrail('a1', 1) == 'a1-h8');
  assert(mirrorSquareTrail('d5', 2) == 'd5-e4-d5');
  assert(mirrorSquareTrail('', 4) == '');
  print('All tests passed!');
}