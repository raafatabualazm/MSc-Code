@pragma('vm:entry-point')
List<int> dealEvenSuitCards(List<int> grid, int gridWidth, int suitRow) {
  if (gridWidth <= 0 || grid.isEmpty) return [];
  int rowCount = grid.length ~/ gridWidth;
  if (suitRow < 0 || suitRow >= rowCount) return [];
  List<int> result = [];
  for (int col = 0; col < gridWidth; col++) {
    if (col % 2 == 0) {
      result.add(grid[suitRow * gridWidth + col]);
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(dealEvenSuitCards([1,2,3,4,5,6], 3, 0).toString() == '[1, 3]');
  assert(dealEvenSuitCards([1,2,3,4,5,6], 3, 1).toString() == '[4, 6]');
  assert(dealEvenSuitCards([], 3, 0).toString() == '[]');
  print('All tests passed!');
}