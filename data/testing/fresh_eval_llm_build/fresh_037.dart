@pragma('vm:entry-point')
List<double> batteryCycleBandWear(List<int> cycleReadings, int bandSize, double wearStep) {
  final counts = <int, int>{};
  for (final cycles in cycleReadings) {
    final band = cycles ~/ bandSize;
    counts[band] = (counts[band] ?? 0) + 1;
  }
  return [
    (counts[0] ?? 0) * wearStep,
    (counts[1] ?? 0) * wearStep,
    (counts[2] ?? 0) * wearStep
  ];
}

@pragma('vm:entry-point')
void main() {
  assert(batteryCycleBandWear([], 4, 0.5).length == 3);
  assert(batteryCycleBandWear([0, 3, 4, 8], 4, 0.5).toString() == '[1.0, 0.5, 0.5]');
  assert(batteryCycleBandWear([2, 2, 7], 5, 1.0)[1] == 1.0);
  print('All tests passed!');
}