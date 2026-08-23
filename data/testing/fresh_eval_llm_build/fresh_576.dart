@pragma('vm:entry-point')
List<int> auditChessSquareIntervals(List<int> records) {
  List<int> score = [0, 0, 0];
  for (int i = 0; i < records.length; i++) {
    int dayA = records[i] ~/ 64;
    int squareA = records[i] % 64;
    int rowA = squareA ~/ 8;
    int colA = squareA % 8;
    for (int j = i + 1; j < records.length; j++) {
      int dayGap = (records[j] ~/ 64) - dayA;
      if (dayGap <= 0) continue;
      int squareB = records[j] % 64;
      int rowB = squareB ~/ 8;
      int colB = squareB % 8;
      int dr = (rowA - rowB).abs();
      int dc = (colA - colB).abs();
      if (squareA == squareB) {
        score[0] += dayGap;
        if (dayGap > 20) return score;
      } else if (rowA == rowB || colA == colB || dr == dc) {
        score[1] += dayGap.isOdd ? 1 : 2;
      } else if ((dr == 1 && dc == 2) || (dr == 2 && dc == 1)) {
        score[2] += dayGap % 5;
      } else if (dayGap % 7 == 0) {
        score[2]--;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(auditChessSquareIntervals([]).toString() == '[0, 0, 0]');
  assert(auditChessSquareIntervals([0, 64]).toString() == '[1, 0, 0]');
  assert(auditChessSquareIntervals([0, 65, 202]).toString() == '[0, 3, 3]');
  print('All tests passed!');
}