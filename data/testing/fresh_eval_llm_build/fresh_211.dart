@pragma('vm:entry-point')
int countCrowdedWifiBins(List<int> readings) {
  final counts = <int, int>{};
  int crowded = 0;
  for (final r in readings) {
    final bin = (-r) ~/ 10;
    counts[bin] = (counts[bin] ?? 0) + 1;
    if (counts[bin] == 3) crowded++;
  }
  return crowded;
}

@pragma('vm:entry-point')
void main() {
  assert(countCrowdedWifiBins([]) == 0);
  assert(countCrowdedWifiBins([-42, -44, -48]) == 1);
  assert(countCrowdedWifiBins([-49, -50, -51, -52]) == 1);
  print('All tests passed!');
}