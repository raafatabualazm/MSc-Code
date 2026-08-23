@pragma('vm:entry-point')
List<int> tideShiftBands(List<List<int>> readings) {
  List<int> bands = [];
  for (int r = 0; r < readings.length; r++) {
    int score = 0;
    for (int c = 0; c < readings[r].length; c++) {
      int v = readings[r][c];
      if (c > 0) {
        int delta = v - readings[r][c - 1];
        if (delta >= 3) {
          score += 2;
        } else if (delta <= -3) {
          score -= 2;
        } else if (delta != 0) {
          score += delta > 0 ? 1 : -1;
        }
      }
      if (r > 0 && c < readings[r - 1].length && readings[r - 1][c] == v) {
        score++;
      }
    }
    bands.add(score);
  }
  return bands;
}

@pragma('vm:entry-point')
void main() {
  assert(tideShiftBands([]).toString() == '[]');
  assert(tideShiftBands([[1, 4]]).toString() == '[2]');
  assert(tideShiftBands([[2, 5, 2], [2, 2]]).toString() == '[0, 1]');
  print('All tests passed!');
}