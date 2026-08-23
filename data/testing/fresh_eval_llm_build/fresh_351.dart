@pragma('vm:entry-point')
List<int> markDryRows(List<List<int>> grid) {
  List<int> result = [];
  for (var row in grid) {
    if (row.isEmpty) {
      result.add(0);
      continue;
    }
    int minMoisture = row.reduce((a,b) => a < b ? a : b);
    result.add(minMoisture < 15 ? 1 : 0);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(markDryRows([]).toString() == '[]');
  assert(markDryRows([[14]]).toString() == '[1]');
  assert(markDryRows([[15]]).toString() == '[0]');
  print('All tests passed!');
}