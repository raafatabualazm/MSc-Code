@pragma('vm:entry-point')
int maxRowSignatureGroup(List<String> rows) {
  Map<String, int> groupSizes = {};
  for (String row in rows) {
    int hashCount = 0;
    int dotCount = 0;
    for (int i = 0; i < row.length; i++) {
      if (row[i] == '#') {
        hashCount++;
      } else if (row[i] == '.') {
        dotCount++;
      }
    }
    String key = 'h$hashCount d$dotCount';
    groupSizes[key] = (groupSizes[key] ?? 0) + 1;
  }
  int maxSize = 0;
  for (int count in groupSizes.values) {
    if (count > maxSize) {
      maxSize = count;
    }
  }
  return maxSize;
}

@pragma('vm:entry-point')
void main() {
  assert(maxRowSignatureGroup([]) == 0);
  assert(maxRowSignatureGroup(['#', '.']) == 1);
  assert(maxRowSignatureGroup(['#.', '.#']) == 2);
  print('All tests passed!');
}