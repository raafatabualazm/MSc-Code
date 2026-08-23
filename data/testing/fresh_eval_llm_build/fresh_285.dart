@pragma('vm:entry-point')
String? summarizeThermostatAnomalies(List<List<int>> schedule) {
  if (schedule.isEmpty) return null;
  int width = schedule[0].length;
  if (width == 0) return 'empty-day';
  int spikes = 0;
  int calmRows = 0;
  for (int r = 0; r < schedule.length; r++) {
    List<int> row = schedule[r];
    if (row.length != width) return 'ragged';
    int rowChanges = 0;
    bool allSame = true;
    for (int c = 0; c < row.length; c++) {
      int v = row[c];
      if (v < 10 || v > 30) return 'unsafe:$r,$c';
      if (c > 0) {
        int diff = (v - row[c - 1]).abs();
        if (diff >= 5) {
          spikes++;
          rowChanges++;
        }
        if (v != row[c - 1]) allSame = false;
      }
      if (r > 0 && (v - schedule[r - 1][c]).abs() >= 7) {
        spikes += 2;
        continue;
      }
      if (c > 1 && v == row[c - 2] && v != row[c - 1]) rowChanges++;
    }
    if (allSame) calmRows++;
    if (rowChanges > 2) return 'oscillating:$r';
  }
  if (spikes == 0) return 'stable:$calmRows';
  return 'spikes:$spikes:$calmRows';
}

@pragma('vm:entry-point')
void main() {
  assert(summarizeThermostatAnomalies([]) == null);
  assert(summarizeThermostatAnomalies([[15, 15], [15, 15]]) == 'stable:2');
  assert(summarizeThermostatAnomalies([[15, 20, 15]]) == 'oscillating:0');
  print('All tests passed!');
}