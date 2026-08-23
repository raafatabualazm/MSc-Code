@pragma('vm:entry-point')
List<int> firstModerateMoistureIndices(String grid) {
  if (grid.isEmpty) return [];
  final rows = grid.split(';');
  final result = <int>[];
  for (final row in rows) {
    final cells = row.split(',');
    final idx = cells.indexWhere((c) {
      final v = int.tryParse(c);
      return v != null && v >= 30 && v <= 60;
    });
    result.add(idx);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(firstModerateMoistureIndices("").toString() == "[]");
  assert(firstModerateMoistureIndices("30,45;50,60").toString() == "[0, 0]");
  assert(firstModerateMoistureIndices("29,30,61").toString() == "[1]");
  print('All tests passed!');
}