@pragma('vm:entry-point')
int computeManifestBalanceScore(List<List<int>> itemPositions, List<int> referencePoint) {
  int total = 0;
  for (var p in itemPositions) {
    int d = (p[0] - referencePoint[0]).abs() + (p[1] - referencePoint[1]).abs();
    total += d.isEven ? d : -d;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(computeManifestBalanceScore([], [0, 0]) == 0);
  assert(computeManifestBalanceScore([[3, 4]], [1, 2]) == 4);
  assert(computeManifestBalanceScore([[5, 5], [6, 6]], [5, 5]) == 2);
  print('All tests passed!');
}