@pragma('vm:entry-point')
bool hasBorderMoistureSink(List<List<int>> grid) {
  final rows = grid.length;
  if (rows == 0) return false;
  final cols = grid[0].length;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
        bool isMin = true;
        bool hasNeighbor = false;
        if (i > 0) { hasNeighbor = true; if (grid[i][j] >= grid[i-1][j]) isMin = false; }
        if (i < rows - 1) { hasNeighbor = true; if (grid[i][j] >= grid[i+1][j]) isMin = false; }
        if (j > 0) { hasNeighbor = true; if (grid[i][j] >= grid[i][j-1]) isMin = false; }
        if (j < cols - 1) { hasNeighbor = true; if (grid[i][j] >= grid[i][j+1]) isMin = false; }
        if (hasNeighbor && isMin) return true;
      }
    }
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(hasBorderMoistureSink([[3, 1, 4]]) == true);
  assert(hasBorderMoistureSink([[5, 5], [5, 5]]) == false);
  assert(hasBorderMoistureSink([]) == false);
  print('All tests passed!');
}