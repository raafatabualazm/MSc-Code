@pragma('vm:entry-point')
int enclosedSeatGapArea(List<int> reservedSeats, int seatsPerRow) {
  if (reservedSeats.isEmpty) return 0;
  int minRow = 1 << 30, maxRow = -1, minCol = 1 << 30, maxCol = -1;
  for (final seat in reservedSeats) {
    int row = seat ~/ seatsPerRow, col = seat % seatsPerRow;
    if (row < minRow) minRow = row;
    if (row > maxRow) maxRow = row;
    if (col < minCol) minCol = col;
    if (col > maxCol) maxCol = col;
  }
  return (maxRow - minRow + 1) * (maxCol - minCol + 1) - reservedSeats.length;
}

@pragma('vm:entry-point')
void main() {
  assert(enclosedSeatGapArea([], 12) == 0);
  assert(enclosedSeatGapArea([0, 11], 10) == 2);
  assert(enclosedSeatGapArea([9, 10], 10) == 18);
  print('All tests passed!');
}