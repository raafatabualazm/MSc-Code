@pragma('vm:entry-point')
String summarizeTideGrid(List<List<int>> readings) {
  if (readings.isEmpty || readings[0].isEmpty) return 'no-data';
  int basins = 0;
  int crestRows = 0;
  int unstableEdges = 0;
  for (int r = 0; r < readings.length; r++) {
    bool crest = readings[r].length > 1;
    for (int c = 0; c < readings[r].length; c++) {
      int v = readings[r][c];
      if ((r == 0 || c == 0 || r == readings.length - 1 || c == readings[r].length - 1) && v < 0) {
        unstableEdges++;
      } else if (r > 0 && c > 0 && r < readings.length - 1 && c < readings[r].length - 1 &&
          v < readings[r - 1][c] && v < readings[r + 1][c] &&
          v < readings[r][c - 1] && v < readings[r][c + 1]) {
        basins++;
      }
      if (c > 0 && v - readings[r][c - 1] != 1) crest = false;
    }
    if (crest) crestRows++;
  }
  return 'B$basins-C$crestRows-E$unstableEdges';
}

@pragma('vm:entry-point')
void main() {
  assert(summarizeTideGrid([]) == 'no-data');
  assert(summarizeTideGrid([[0, 1, 2], [3, 0, 5], [2, 3, 4]]) == 'B1-C2-E0');
  assert(summarizeTideGrid([[-2, -1, 0, 1]]) == 'B0-C1-E2');
  print('All tests passed!');
}