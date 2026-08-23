@pragma('vm:entry-point')
List<String> orderCyclesByDepth(List<String> cycles) {
  cycles.sort((a, b) {
    var partsA = a.split('-');
    var partsB = b.split('-');
    int startA = int.parse(partsA[0]);
    int endA = int.parse(partsA[1]);
    int startB = int.parse(partsB[0]);
    int endB = int.parse(partsB[1]);
    int depthA = startA - endA;
    int depthB = startB - endB;
    int cmp = depthB - depthA; // descending depth
    if (cmp != 0) return cmp;
    cmp = startB - startA; // descending start
    if (cmp != 0) return cmp;
    return a.compareTo(b);
  });
  return cycles;
}

@pragma('vm:entry-point')
void main() {
  assert(orderCyclesByDepth([]).isEmpty);
  assert(orderCyclesByDepth(["100-80"]).first == "100-80");
  assert(orderCyclesByDepth(["90-70", "100-80"]).toString() == "[100-80, 90-70]");
  print('All tests passed!');
}