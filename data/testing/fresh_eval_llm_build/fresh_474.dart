@pragma('vm:entry-point')
int countPhaseInversions(List<String> phases, List<String> priorityOrder) {
  if (priorityOrder.isEmpty) return 0;
  Map<String, int> priorityMap = {};
  for (int i = 0; i < priorityOrder.length; i++) {
    priorityMap[priorityOrder[i]] = i;
  }
  int defaultPriority = priorityOrder.length;
  int inversions = 0;
  for (int i = 0; i < phases.length; i++) {
    int p1 = priorityMap[phases[i]] ?? defaultPriority;
    for (int j = i + 1; j < phases.length; j++) {
      int p2 = priorityMap[phases[j]] ?? defaultPriority;
      if (p1 > p2) {
        inversions++;
      }
    }
  }
  return inversions;
}

@pragma('vm:entry-point')
void main() {
  assert(countPhaseInversions([], ["green","yellow","red"]) == 0);
  assert(countPhaseInversions(["green","red","yellow"], ["green","yellow","red"]) == 1);
  assert(countPhaseInversions(["red","yellow","green"], ["green","yellow","red"]) == 3);
  print('All tests passed!');
}