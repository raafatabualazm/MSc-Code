@pragma('vm:entry-point')
import 'dart:math';

int countAmberGridConflicts(List<String> lights) {
  int total = 0;
  for (int i = 0; i < lights.length; i++) {
    List<String> a = lights[i].split(',');
    if (a.length != 4) continue;
    int x1 = int.parse(a[0]), y1 = int.parse(a[1]);
    int g1 = int.parse(a[2]), a1 = int.parse(a[3]);
    for (int j = i + 1; j < lights.length; j++) {
      List<String> b = lights[j].split(',');
      if (b.length != 4) continue;
      int x2 = int.parse(b[0]), y2 = int.parse(b[1]);
      int g2 = int.parse(b[2]), a2 = int.parse(b[3]);
      int dist = (x1 - x2).abs() + (y1 - y2).abs();
      if (((g1 + a1) & 1) == ((g2 + a2) & 1)) {
        if (dist == 0) total += 1;
        continue;
      }
      if (dist > g1 + g2) continue;
      int left = max(x1 - a1, x2 - a2), right = min(x1 + a1, x2 + a2);
      int bottom = max(y1 - a1, y2 - a2), top = min(y1 + a1, y2 + a2);
      if (left > right || bottom > top) {
        if (x1 == x2 || y1 == y2) total += dist;
        continue;
      }
      int area = (right - left + 1) * (top - bottom + 1);
      total += dist < a1 + a2 ? area * 2 : area;
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(countAmberGridConflicts([]) == 0);
  assert(countAmberGridConflicts(['0,0,1,1','0,0,3,1']) == 1);
  assert(countAmberGridConflicts(['0,0,2,1','0,1,2,2','1,1,1,0']) == 20);
  print('All tests passed!');
}