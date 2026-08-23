@pragma('vm:entry-point')
List<String> describeQrModuleRuns(List<String> grid) {
  List<String> out = [];
  if (grid.isEmpty) {
    return ['empty'];
  }
  for (int r = 0; r < grid.length; r++) {
    String row = grid[r];
    if (row.isEmpty) {
      out.add('row${r}:blank');
      continue;
    }
    int c = 0;
    while (c < row.length) {
      String ch = row[c];
      if (ch != '#' && ch != '.') {
        c++;
        continue;
      }
      int start = c;
      while (c < row.length && row[c] == ch) {
        c++;
      }
      int len = c - start;
      int aligned = 0;
      if (r > 0) {
        for (int k = start; k < c && k < grid[r - 1].length; k++) {
          if (grid[r - 1][k] == ch) {
            aligned++;
          }
        }
      }
      if (ch == '#') {
        if ((start == 0 || c == row.length) && len >= 3) {
          out.add('F${r}_$len');
        } else if (aligned >= 2) {
          out.add('A${r}_$len');
        } else {
          out.add('D${r}_$len');
        }
      } else {
        if (len >= 4) {
          continue;
        }
        out.add(aligned == len ? 'Q${r}_$len' : 'L${r}_$len');
      }
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(describeQrModuleRuns([]).toString() == '[empty]');
  assert(describeQrModuleRuns(['###']).toString() == '[F0_3]');
  assert(describeQrModuleRuns(['..', '..']).toString() == '[L0_2, Q1_2]');
  print('All tests passed!');
}