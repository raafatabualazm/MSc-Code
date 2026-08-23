@pragma('vm:entry-point')
bool isSoilMoistureBalanced(List<List<int>> grid) {
  const int threshold = 20;
  int n = grid.length;
  if (n == 0) return false;
  for (var row in grid) {
    if (row.length != n) return false;
  }
  for (int i = 0; i < n; i++) {
    int rowSum = 0;
    int colSum = 0;
    for (int j = 0; j < n; j++) {
      int valRow = grid[i][j];
      if (valRow >= threshold) rowSum += valRow;
      int valCol = grid[j][i];
      if (valCol >= threshold) colSum += valCol;
    }
    if (rowSum != colSum) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isSoilMoistureBalanced([[20, 0], [0, 20]]) == true);
  assert(isSoilMoistureBalanced([[20, 0], [20, 0]]) == false);
  assert(isSoilMoistureBalanced([]) == false);
  print('All tests passed!');
}