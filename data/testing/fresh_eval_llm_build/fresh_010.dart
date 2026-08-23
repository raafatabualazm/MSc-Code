@pragma('vm:entry-point')
import 'dart:math';

int? mostContextuallyOverlappingSong(List<List<int>> songZones) {
  int n = songZones.length;
  if (n == 0) return null;
  if (n == 1) return 0;
  int bestIndex = 0;
  int bestTotal = -1;
  for (int i = 0; i < n; i++) {
    int total = 0;
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      int xOverlap = min(songZones[i][2], songZones[j][2]) - max(songZones[i][0], songZones[j][0]);
      if (xOverlap <= 0) continue;
      int yOverlap = min(songZones[i][3], songZones[j][3]) - max(songZones[i][1], songZones[j][1]);
      if (yOverlap <= 0) continue;
      total += xOverlap * yOverlap;
    }
    if (total > bestTotal) {
      bestTotal = total;
      bestIndex = i;
    }
  }
  return bestIndex;
}

@pragma('vm:entry-point')
void main() {
  assert(mostContextuallyOverlappingSong([]) == null);
  assert(mostContextuallyOverlappingSong([[1,1,2,2], [1,1,2,2]]) == 0);
  assert(mostContextuallyOverlappingSong([[0,0,10,10], [1,1,2,2], [2,2,3,3]]) == 0);
  print('All tests passed!');
}