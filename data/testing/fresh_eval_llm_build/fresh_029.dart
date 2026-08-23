@pragma('vm:entry-point')
int countMazeOpenCells(int depth) {
  if (depth <= 0) return 0;
  if (depth == 1) return 1;
  return 2 * countMazeOpenCells(depth - 1) + depth;
}

@pragma('vm:entry-point')
void main() {
  assert(countMazeOpenCells(1) == 1);
  assert(countMazeOpenCells(3) == 11);
  assert(countMazeOpenCells(0) == 0);
  print('All tests passed!');
}