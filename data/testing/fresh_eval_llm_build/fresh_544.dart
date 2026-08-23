@pragma('vm:entry-point')
List<int> summarizeBucketStripePressure(List<List<int>> buckets, int pivot) {
  if (buckets.isEmpty) return [];
  List<int> result = [];
  for (int r = 0; r < buckets.length; r++) {
    List<int> row = buckets[r];
    if (row.isEmpty) {
      result.add(0);
      continue;
    }
    int pressure = 0;
    for (int c = 0; c < row.length; c++) {
      int v = row[c];
      if (v < 0) continue;
      if (v == pivot) {
        pressure += 2;
      } else if (v > pivot) {
        pressure += v - pivot;
        if (c > 0 && row[c - 1] > v) pressure--;
      } else {
        pressure -= pivot - v;
        if (c == 0 || c == row.length - 1) pressure++;
      }
      if (c > 0 && row[c - 1] == v) pressure++;
      if (r > 0 && c < buckets[r - 1].length) {
        int up = buckets[r - 1][c];
        if ((up - v).abs() == 1) {
          pressure++;
        } else if (up == v) {
          pressure--;
        }
      }
    }
    result.add(pressure < 0 ? 0 : pressure);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(summarizeBucketStripePressure([], 3).toString() == '[]');
  assert(summarizeBucketStripePressure([[3, 2], [2, 3]], 3).toString() == '[2, 4]');
  assert(summarizeBucketStripePressure([[4, 4], [4, 4]], 3).toString() == '[3, 1]');
  print('All tests passed!');
}