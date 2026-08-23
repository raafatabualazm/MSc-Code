@pragma('vm:entry-point')
List<int> packetFrameCarryLevels(List<int> packetSizes, int frameLimit) {
  List<int> levels = [];
  int current = 0;
  for (final size in packetSizes) {
    current = size > frameLimit ? 0 : (current + size > frameLimit ? size : current + size);
    levels.add(current);
  }
  return levels;
}

@pragma('vm:entry-point')
void main() {
  assert(packetFrameCarryLevels([], 8).toString() == '[]');
  assert(packetFrameCarryLevels([6, 5], 10).toString() == '[6, 5]');
  assert(packetFrameCarryLevels([3, 3, 3, 3], 6).toString() == '[3, 6, 3, 6]');
  print('All tests passed!');
}