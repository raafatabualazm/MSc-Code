@pragma('vm:entry-point')
int? collapsedServerZoneScore(List<String> logs, int limit) {
  final rects = <List<int>>[];
  for (final line in logs) {
    final parts = line.split(',');
    if (parts.length != 4) continue;
    final values = <int>[];
    var ok = true;
    for (final part in parts) {
      final v = int.tryParse(part.trim());
      if (v == null) {
        ok = false;
        break;
      }
      values.add(v);
    }
    if (!ok || values[2] <= 0 || values[3] <= 0) continue;
    rects.add(values);
  }
  if (rects.isEmpty) return null;
  var score = 0;
  for (var i = 0; i < rects.length; i++) {
    for (var j = i + 1; j < rects.length; j++) {
      final a = rects[i], b = rects[j];
      final dist = (a[0] - b[0]).abs() + (a[1] - b[1]).abs();
      if (dist > limit) continue;
      final ow = ((a[0] + a[2]) < (b[0] + b[2]) ? (a[0] + a[2]) : (b[0] + b[2])) - (a[0] > b[0] ? a[0] : b[0]);
      final oh = ((a[1] + a[3]) < (b[1] + b[3]) ? (a[1] + a[3]) : (b[1] + b[3])) - (a[1] > b[1] ? a[1] : b[1]);
      if (ow <= 0 || oh <= 0) {
        if (dist == limit) score--;
        continue;
      }
      final area = ow * oh;
      if ((a[0] < 0) != (b[0] < 0) && (a[1] < 0) != (b[1] < 0)) {
        score += area * 2;
      } else if (dist == 0) {
        score += area + limit;
      } else {
        score += area;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(collapsedServerZoneScore([], 3) == null);
  assert(collapsedServerZoneScore(['0,0,3,3','1,1,3,2'], 3) == 4);
  assert(collapsedServerZoneScore(['0,0,2,2','2,0,2,2'], 2) == -1);
  print('All tests passed!');
}