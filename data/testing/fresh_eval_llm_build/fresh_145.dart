@pragma('vm:entry-point')
String analyzeIntersectionConflicts(List<List<int>> intersections) {
  if (intersections.isEmpty) {
    return "Conflicts:0, MaxConflict: none";
  }
  int n = intersections.length;
  if (n == 1) {
    return "Conflicts:0, MaxConflict: (${intersections[0][0]},${intersections[0][1]}) with 0";
  }
  int conflictCount = 0;
  List<int> conflictsPer = List.filled(n, 0);
  const int range = 3;
  
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      int dx = (intersections[i][0] - intersections[j][0]).abs();
      int dy = (intersections[i][1] - intersections[j][1]).abs();
      int distance = dx + dy;
      if (distance > range) {
        continue;
      }
      if (intersections[i][2] == intersections[j][2]) {
        continue;
      }
      conflictCount++;
      conflictsPer[i]++;
      conflictsPer[j]++;
    }
  }
  
  int maxIdx = 0;
  for (int i = 1; i < n; i++) {
    if (conflictsPer[i] > conflictsPer[maxIdx]) {
      maxIdx = i;
    }
  }
  
  return "Conflicts:$conflictCount, MaxConflict: (${intersections[maxIdx][0]},${intersections[maxIdx][1]}) with ${conflictsPer[maxIdx]}";
}

@pragma('vm:entry-point')
void main() {
  assert(analyzeIntersectionConflicts([]) == "Conflicts:0, MaxConflict: none");
  assert(analyzeIntersectionConflicts([[0,0,1],[1,1,0]]) == "Conflicts:1, MaxConflict: (0,0) with 1");
  print('All tests passed!');
}