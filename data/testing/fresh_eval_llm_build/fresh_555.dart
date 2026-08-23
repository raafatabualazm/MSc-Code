@pragma('vm:entry-point')
bool validateTideBasins(List<List<int>> readings) {
  if (readings.isEmpty) return true;
  int width = readings[0].length;
  if (width == 0) return false;
  int basins = 0;
  Set<String> basinMarks = {};
  for (int i = 0; i < readings.length; i++) {
    if (readings[i].length != width) return false;
    int trend = 0;
    int reversals = 0;
    for (int j = 0; j < width; j++) {
      if (j > 0) {
        int diff = readings[i][j] - readings[i][j - 1];
        if (diff.abs() > 2) return false;
        int dir = diff == 0 ? 0 : (diff > 0 ? 1 : -1);
        if (dir != 0) {
          if (trend != 0 && dir != trend) reversals++;
          trend = dir;
        }
        if (reversals > 1) return false;
      }
      if (i > 0 && i < readings.length - 1 && j > 0 && j < width - 1) {
        int v = readings[i][j];
        if (v < readings[i - 1][j] && v < readings[i + 1][j] && v < readings[i][j - 1] && v < readings[i][j + 1]) {
          if (v.isOdd) return false;
          basins++;
          if (basinMarks.contains('${i - 1}:${j - 1}') || basinMarks.contains('${i - 1}:${j + 1}')) return false;
          basinMarks.add('$i:$j');
        } else if (v > readings[i - 1][j] && v > readings[i + 1][j] && v > readings[i][j - 1] && v > readings[i][j + 1] && v < 0) {
          return false;
        }
      }
    }
  }
  return readings.length == 1 || width == 1 ? basins == 0 : basins > 0;
}

@pragma('vm:entry-point')
void main() {
  assert(validateTideBasins([]) == true);
  assert(validateTideBasins([[2, 1, 2], [1, 0, 1], [2, 1, 2]]) == true);
  assert(validateTideBasins([[1, 2, 1, 2]]) == false);
  print('All tests passed!');
}