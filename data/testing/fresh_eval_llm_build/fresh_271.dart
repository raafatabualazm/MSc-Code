@pragma('vm:entry-point')
int longPhaseTotalTime(List<String> phaseStrings, int minTime) {
  int total = 0;
  for (final s in phaseStrings) {
    if (s.length >= 2) {
      String numStr = s.substring(1);
      int duration = int.tryParse(numStr) ?? 0;
      if (duration >= minTime) {
        total += duration;
      }
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(longPhaseTotalTime([], 10) == 0);
  assert(longPhaseTotalTime(["R5", "Y3", "G20"], 5) == 25);
  assert(longPhaseTotalTime(["R7", "Y7", "G7"], 7) == 21);
  print('All tests passed!');
}