@pragma('vm:entry-point')
List<int> parseTrafficLightCycles(String phases) {
  // States: 0=waiting for R, 1=got R, 2=got G, 3=got Y (cycle complete)
  // Durations: R=30, G=45, Y=5
  final durations = {'R': 30, 'G': 45, 'Y': 5};
  final List<int> results = [];
  int state = 0;
  int cycleSum = 0;
  for (int i = 0; i < phases.length; i++) {
    final ch = phases[i];
    if (ch == 'X') {
      state = 0;
      cycleSum = 0;
      continue;
    }
    if (ch != 'R' && ch != 'G' && ch != 'Y') {
      continue;
    }
    if (state == 0) {
      if (ch == 'R') { state = 1; cycleSum = 30; }
    } else if (state == 1) {
      if (ch == 'G') { state = 2; cycleSum += 45; }
      else if (ch == 'R') { cycleSum = 30; }
      else { state = 0; cycleSum = 0; }
    } else if (state == 2) {
      if (ch == 'Y') { state = 3; cycleSum += 5; results.add(cycleSum); cycleSum = 0; state = 0; }
      else if (ch == 'G') { cycleSum += 45; }
      else { state = 0; cycleSum = 0; }
    }
  }
  return results;
}

@pragma('vm:entry-point')
void main() {
  assert(parseTrafficLightCycles('').toString() == '[]');
  assert(parseTrafficLightCycles('RGY').toString() == '[80]');
  assert(parseTrafficLightCycles('RGYRGY').toString() == '[80, 80]');
  print('All tests passed!');
}