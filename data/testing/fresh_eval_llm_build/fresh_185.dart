@pragma('vm:entry-point')
bool allIngredientsScaleWithinColumnBounds(List<List<int>> grid, int scaleFactor) {
  if (grid.isEmpty) return true;
  int rows = grid.length;
  int cols = grid[0].length;
  if (cols == 0) return true;
  // Compute column maxima
  List<int> colMax = List<int>.filled(cols, 0);
  for (int r = 0; r < rows; r++) {
    if (grid[r].length != cols) return false;
    for (int c = 0; c < cols; c++) {
      if (grid[r][c] > colMax[c]) colMax[c] = grid[r][c];
    }
  }
  // Check each scaled value against column max
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      int val = grid[r][c];
      if (val < 0) return false;
      int scaled = val * scaleFactor;
      if (scaled > colMax[c]) return false;
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(allIngredientsScaleWithinColumnBounds([[1,2],[3,4]], 1) == true);
  assert(allIngredientsScaleWithinColumnBounds([[2,4],[3,1]], 2) == false);
  assert(allIngredientsScaleWithinColumnBounds([], 5) == true);
  print('All tests passed!');
}