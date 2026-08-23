@pragma('vm:entry-point')
bool asciiRectBandCovered(List<String> rows, int rectCode) {
  int width = rectCode ~/ 1000;
  int height = rectCode % 1000;
  if (rows.isEmpty || height <= 0 || width <= 0 || height > rows.length) return false;
  for (int start = 0; start <= rows.length - height; start++) {
    bool bandCovered = true;
    for (int r = start; r < start + height; r++) {
      bool rowHit = false;
      for (int c = 0; c < width && c < rows[r].length; c++) {
        if (rows[r][c] == '#') { rowHit = true; break; }
      }
      if (!rowHit) { bandCovered = false; break; }
    }
    if (bandCovered) return true;
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(asciiRectBandCovered(["..#..", "##...", ".....", "..#.."], 3002) == true);
  assert(asciiRectBandCovered([".....", ".....", "....#"], 3001) == false);
  assert(asciiRectBandCovered([], 3001) == false);
  print('All tests passed!');
}