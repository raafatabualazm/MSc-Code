@pragma('vm:entry-point')
List<int> collectPhaseDurations(String phaseLog, String phase) {
  List<int> durations = [];
  for (String token in phaseLog.split('|')) {
    if (token.isNotEmpty && token.startsWith(phase)) {
      durations.add(int.parse(token.substring(1)));
    }
  }
  return durations;
}

@pragma('vm:entry-point')
void main() {
  assert(collectPhaseDurations('G5|Y2|G7', 'G').toString() == '[5, 7]');
  assert(collectPhaseDurations('', 'R').length == 0);
  assert(collectPhaseDurations('R0|R1|G3', 'R')[0] == 0);
  print('All tests passed!');
}