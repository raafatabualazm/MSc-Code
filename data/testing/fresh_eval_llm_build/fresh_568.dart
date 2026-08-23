@pragma('vm:entry-point')
String mazeCellGeometryReport(List<String> cells) {
  if (cells.isEmpty) return 'empty';
  final seen = <String>{};
  final points = <List<int>>[];
  int minX = 1 << 30, minY = 1 << 30;
  int maxX = -(1 << 30), maxY = -(1 << 30);
  int overlaps = 0;
  for (final cell in cells) {
    final parts = cell.split(':');
    if (parts.length != 2) continue;
    final x = int.tryParse(parts[0]);
    final y = int.tryParse(parts[1]);
    if (x == null || y == null) continue;
    final key = '$x:$y';
    if (!seen.add(key)) {
      overlaps++;
      continue;
    }
    points.add([x, y]);
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  if (points.isEmpty) return 'empty';
  int touching = 0;
  int isolated = 0;
  for (int i = 0; i < points.length; i++) {
    bool linked = false;
    for (int j = 0; j < points.length; j++) {
      if (i == j) continue;
      final d = (points[i][0] - points[j][0]).abs() +
          (points[i][1] - points[j][1]).abs();
      if (d == 1) {
        if (i < j) touching++;
        linked = true;
      } else if (d == 2 &&
          (points[i][0] == points[j][0] ||
              points[i][1] == points[j][1])) {
        linked = true;
      }
    }
    if (!linked) isolated++;
  }
  final area = (maxX - minX + 1) * (maxY - minY + 1);
  return 'A$area-T$touching-O$overlaps-I$isolated';
}

@pragma('vm:entry-point')
void main() {
  assert(mazeCellGeometryReport([]) == 'empty');
  assert(mazeCellGeometryReport(['0:0', '1:0']) == 'A2-T1-O0-I0');
  assert(mazeCellGeometryReport(['0:0', '0:0']) == 'A1-T0-O1-I1');
  print('All tests passed!');
}