@pragma('vm:entry-point')
List<String> flagPasswordResetIntervals(List<int> resetDays) {
  List<String> flags = [];
  for (int i = 1; i < resetDays.length; i++) {
    int gap = resetDays[i] - resetDays[i - 1];
    if (gap < 7 || gap > 60) {
      flags.add('gap$i:$gap');
    }
  }
  return flags;
}

@pragma('vm:entry-point')
void main() {
  assert(flagPasswordResetIntervals([]).length == 0);
  assert(flagPasswordResetIntervals([10, 16]).toString() == '[gap1:6]');
  assert(flagPasswordResetIntervals([0, 7, 67]).length == 0);
  print('All tests passed!');
}