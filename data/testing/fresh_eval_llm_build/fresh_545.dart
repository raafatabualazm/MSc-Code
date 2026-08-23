@pragma('vm:entry-point')
List<List<int>> decodeShiftedChessBursts(String tape) {
  List<List<int>> out = [];
  if (tape.isEmpty) return out;
  for (String part in tape.split('|')) {
    if (part.isEmpty) continue;
    int i = 0;
    int repeat = 0;
    while (i < part.length && part.codeUnitAt(i) >= 48 && part.codeUnitAt(i) <= 57) {
      repeat = repeat * 10 + (part.codeUnitAt(i) - 48);
      i++;
    }
    if (repeat == 0) repeat = 1;
    if (i + 1 >= part.length) continue;
    int fileCode = part.codeUnitAt(i);
    int rankCode = part.codeUnitAt(i + 1);
    if (fileCode < 97 || fileCode > 104 || rankCode < 49 || rankCode > 56) continue;
    int file = (fileCode - 97 - repeat) % 8;
    int rank = (rankCode - 49 - repeat) % 8;
    int checksum = 0;
    for (int j = i + 2; j < part.length; j++) {
      checksum += part.codeUnitAt(j);
    }
    if (checksum.isOdd) file = (file + 1) % 8;
    for (int k = 0; k < repeat; k++) {
      if (checksum > 0 && checksum % 3 == 0 && k.isOdd) continue;
      out.add([file < 0 ? file + 8 : file, rank < 0 ? rank + 8 : rank]);
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(decodeShiftedChessBursts('').toString() == '[]');
  assert(decodeShiftedChessBursts('2c3').toString() == '[[0, 0], [0, 0]]');
  assert(decodeShiftedChessBursts('1b2a').toString() == '[[1, 0]]');
  print('All tests passed!');
}