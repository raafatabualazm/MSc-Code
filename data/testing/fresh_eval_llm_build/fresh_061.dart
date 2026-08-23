@pragma('vm:entry-point')
List<int> decodeMoistureGridRow(String encoded, int threshold) {
  if (encoded.isEmpty) return [];
  final List<int> result = [];
  for (final segment in encoded.split(',')) {
    final parts = segment.split(':');
    final count = int.parse(parts[0]);
    final level = int.parse(parts[1]);
    final value = level < threshold ? 0 : level;
    for (int i = 0; i < count; i++) result.add(value);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(decodeMoistureGridRow('3:45,2:12,1:78', 20).toString() == '[45, 45, 45, 0, 0, 78]');
  assert(decodeMoistureGridRow('', 10).isEmpty);
  assert(decodeMoistureGridRow('4:50', 51).toString() == '[0, 0, 0, 0]');
  print('All tests passed!');
}