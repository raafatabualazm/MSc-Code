@pragma('vm:entry-point')
List<int> circularManhattanDistances(List<List<int>> positions) {
  int n = positions.length;
  if (n == 0) return [];
  List<int> result = List.filled(n, 0);
  for (int i = 0; i < n; i++) {
    var p1 = positions[i];
    var p2 = positions[(i + 1) % n];
    int dx = (p1[0] - p2[0]).abs();
    int dy = (p1[1] - p2[1]).abs();
    result[i] = dx + dy;
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(circularManhattanDistances([[0,0],[3,4]]).toString() == '[7, 7]');
  assert(circularManhattanDistances([[1,2],[4,6],[0,0]]).toString() == '[7, 10, 3]');
  assert(circularManhattanDistances([]).length == 0);
  print('All tests passed!');
}