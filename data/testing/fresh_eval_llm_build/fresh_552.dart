@pragma('vm:entry-point')
bool packetsAreManhattanDisjoint(List<List<int>> packets) {
  if (packets.isEmpty) return true;
  for (var p in packets) {
    if (p.length != 3 || p[2] <= 0) {
      return false;
    }
  }
  for (var i = 0; i < packets.length; i++) {
    var p1 = packets[i];
    int x1 = p1[0], y1 = p1[1], r1 = p1[2];
    int x1min = x1 - r1, x1max = x1 + r1;
    int y1min = y1 - r1, y1max = y1 + r1;
    for (var j = i + 1; j < packets.length; j++) {
      var p2 = packets[j];
      int x2 = p2[0], y2 = p2[1], r2 = p2[2];
      int x2min = x2 - r2, x2max = x2 + r2;
      int y2min = y2 - r2, y2max = y2 + r2;
      if (x1max < x2min || x2max < x1min) continue;
      if (y1max < y2min || y2max < y1min) continue;
      if ((x1 - x2).abs() + (y1 - y2).abs() <= r1 + r2) {
        return false;
      }
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(packetsAreManhattanDisjoint([]) == true);
  assert(packetsAreManhattanDisjoint([[0, 0, 1]]) == true);
  assert(packetsAreManhattanDisjoint([[0, 0, 2], [2, 2, 2]]) == false);
  print('All tests passed!');
}