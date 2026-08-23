@pragma('vm:entry-point')
List<int> scanMazeRowOpenRuns(String row) {
  List<int> runs = [];
  for (int i = 0; i < row.length; i++) {
    if (row[i] == '.') {
      if (i == 0 || row[i - 1] != '.') {
        runs.add(1);
      } else {
        runs.last++;
      }
    }
  }
  return runs;
}

@pragma('vm:entry-point')
void main() {
  assert(scanMazeRowOpenRuns("").toString() == "[]");
  assert(scanMazeRowOpenRuns(".").toString() == "[1]");
  assert(scanMazeRowOpenRuns("#..##..#.").toString() == "[2, 2, 1]");
  print('All tests passed!');
}