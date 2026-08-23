@pragma('vm:entry-point')
List<int> moisturePriorityOrder(List<List<int>> grid) {
  if (grid.isEmpty || grid[0].isEmpty) return [];
  int rows = grid.length;
  int cols = grid[0].length;
  List<List<int>> cells = [];
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      cells.add([i * cols + j, grid[i][j]]);
    }
  }
  cells.sort((a, b) {
    int catA = a[1] < 20 ? 1 : a[1] < 50 ? 2 : 3;
    int catB = b[1] < 20 ? 1 : b[1] < 50 ? 2 : 3;
    if (catA != catB) {
      return catA.compareTo(catB);
    }
    if (a[1] != b[1]) {
      if (catA == 1) return a[1].compareTo(b[1]);
      if (catA == 2) return b[1].compareTo(a[1]);
      return a[1].compareTo(b[1]);
    }
    return a[0].compareTo(b[0]);
  });
  return cells.map((e) => e[0]).toList();
}

@pragma('vm:entry-point')
void main() {
  assert(moisturePriorityOrder([]).toString() == '[]');
  assert(moisturePriorityOrder([[50]]).toString() == '[0]');
  assert(moisturePriorityOrder([[19,80],[30,10]]).toString() == '[3, 0, 2, 1]');
  print('All tests passed!');
}