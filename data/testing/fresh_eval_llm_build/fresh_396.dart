@pragma('vm:entry-point')
String sortMazeCells(List<String> cells) {
  cells.sort((a, b) {
    var aP = a.split(','), bP = b.split(',');
    int aX = int.parse(aP[0]), aY = int.parse(aP[1]), bX = int.parse(bP[0]), bY = int.parse(bP[1]);
    String aT = aP[2], bT = bP[2];
    int aO = aT == 'S' ? 0 : aT == 'E' ? 1 : aT == 'P' ? 2 : 3;
    int bO = bT == 'S' ? 0 : bT == 'E' ? 1 : bT == 'P' ? 2 : 3;
    if (aO != bO) return aO.compareTo(bO);
    int dA = aX * aX + aY * aY, dB = bX * bX + bY * bY;
    if (dA != dB) return dA.compareTo(dB);
    if (aX != bX) return aX.compareTo(bX);
    return aY.compareTo(bY);
  });
  return cells.join(' ');
}

@pragma('vm:entry-point')
void main() {
  assert(sortMazeCells([]) == "");
  assert(sortMazeCells(["0,0,S"]) == "0,0,S");
  assert(sortMazeCells(["1,1,P", "0,0,S"]) == "0,0,S 1,1,P");
  print('All tests passed!');
}