@pragma('vm:entry-point')
int countIsolatedMazeCells(List<List<int>> cells) {
  if (cells.isEmpty) return 0;
  var set = cells.map((c) => '${c[0]},${c[1]}').toSet();
  int isolated = 0;
  for (var cell in cells) {
    int x = cell[0], y = cell[1];
    if (!(set.contains('${x-1},$y') || set.contains('${x+1},$y') || set.contains('$x,${y-1}') || set.contains('$x,${y+1}'))) {
      isolated++;
    }
  }
  return isolated;
}

@pragma('vm:entry-point')
void main() {
  assert(countIsolatedMazeCells([]) == 0);
  assert(countIsolatedMazeCells([[0,0]]) == 1);
  assert(countIsolatedMazeCells([[0,0],[0,1]]) == 0);
  print('All tests passed!');
}