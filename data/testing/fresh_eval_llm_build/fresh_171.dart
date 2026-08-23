@pragma('vm:entry-point')
List<String> mapSatellitePassWindows(List<String> grid) {
  if (grid.isEmpty) return [];
  int width = 0;
  for (final row in grid) {
    if (row.length > width) width = row.length;
  }
  List<String> result = [];
  void pushRun(int c, int start, int end) {
    int span = end - start + 1;
    if (span >= 3) {
      result.add('C$c:$start-$end');
    } else if (span == 2) {
      result.add('C$c:pair@$start');
    } else {
      result.add('C$c:blink@$start');
    }
  }

  for (int c = 0; c < width; c++) {
    int start = -1;
    for (int r = 0; r <= grid.length; r++) {
      String ch = r < grid.length && c < grid[r].length ? grid[r][c] : '.';
      if (ch == 'P') {
        bool jammed = (c > 0 && c - 1 < grid[r].length && grid[r][c - 1] == 'S') ||
            (c + 1 < grid[r].length && grid[r][c + 1] == 'S');
        if (jammed) {
          if (start != -1) pushRun(c, start, r - 1);
          start = -1;
          continue;
        }
        if (start == -1) start = r;
      } else if (start != -1) {
        pushRun(c, start, r - 1);
        start = -1;
      }
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(mapSatellitePassWindows([]).toString() == '[]');
  assert(mapSatellitePassWindows(['P.', 'P.']).toString() == '[C0:pair@0]');
  assert(mapSatellitePassWindows(['PS', 'P.', 'P.']).toString() == '[C0:pair@1]');
  print('All tests passed!');
}