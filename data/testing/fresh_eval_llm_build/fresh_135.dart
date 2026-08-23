@pragma('vm:entry-point')
int findPhasePositionInTrafficSequence(List<String> phases, String target) {
  int order(String c) => c == 'green' ? 0 : c == 'yellow' ? 1 : 2;
  var sorted = List<String>.from(phases);
  sorted.sort((a, b) => order(a).compareTo(order(b)));
  for (int i = 0; i < sorted.length; i++) {
    if (sorted[i] == target) return i;
  }
  return -1;
}

@pragma('vm:entry-point')
void main() {
  assert(findPhasePositionInTrafficSequence(["green", "red", "yellow"], "red") == 2);
  assert(findPhasePositionInTrafficSequence([], "green") == -1);
  assert(findPhasePositionInTrafficSequence(["yellow", "green", "green", "yellow"], "yellow") == 2);
  print('All tests passed!');
}