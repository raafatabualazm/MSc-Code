@pragma('vm:entry-point')
bool mazeRelayRegionsHold(List<String> grid) {
  if (grid.isEmpty) return true;
  var seen = <String>{};
  var dirs = const [[1, 0], [-1, 0], [0, 1], [0, -1]];
  for (var r = 0; r < grid.length; r++) {
    for (var c = 0; c < grid[r].length; c++) {
      var key = '$r:$c';
      if (grid[r][c] == '#' || seen.contains(key)) continue;
      var stack = <List<int>>[[r, c]];
      var markers = <String>{};
      var openCount = 0;
      while (stack.isNotEmpty) {
        var cell = stack.removeLast();
        var cr = cell[0], cc = cell[1], ck = '$cr:$cc';
        if (cr < 0 || cr >= grid.length || cc < 0 || cc >= grid[cr].length) continue;
        if (grid[cr][cc] == '#' || seen.contains(ck)) continue;
        seen.add(ck);
        var ch = grid[cr][cc];
        if (ch == '.') {
          openCount++;
        } else {
          if (markers.contains(ch)) return false;
          markers.add(ch);
        }
        for (var d in dirs) {
          stack.add([cr + d[0], cc + d[1]]);
        }
      }
      var size = openCount + markers.length;
      if (markers.isEmpty && size.isOdd) return false;
      if (markers.isNotEmpty && (size.isEven || openCount < markers.length)) return false;
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(mazeRelayRegionsHold([]) == true);
  assert(mazeRelayRegionsHold(['a..']) == true);
  assert(mazeRelayRegionsHold(['a.a..']) == false);
  print('All tests passed!');
}