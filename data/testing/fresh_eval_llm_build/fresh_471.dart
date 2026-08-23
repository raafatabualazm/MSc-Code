@pragma('vm:entry-point')
String classifyTrafficPhasePattern(List<String> phases) {
  Map<String, int> counts = {};
  Set<String> held = {};
  for (int i = 0; i < phases.length; i++) {
    String p = phases[i].toLowerCase();
    if (p == 'red' || p == 'yellow' || p == 'green') {
      counts[p] = (counts[p] ?? 0) + 1;
      if (i > 0 && p == phases[i - 1].toLowerCase()) {
        held.add(p);
      }
    } else if (p == 'flash') {
      held.add('yellow');
    }
  }
  if (counts.isEmpty) return 'no-cycle';
  String best = '';
  int bestCount = -1;
  bool tie = false;
  for (String color in ['green', 'red', 'yellow']) {
    int c = counts[color] ?? 0;
    if (c > bestCount) {
      best = color;
      bestCount = c;
      tie = false;
    } else if (c == bestCount && c > 0) {
      tie = true;
    }
  }
  if (tie) return 'mixed';
  if (bestCount == 1 && counts.length == 1) return 'isolated-' + best;
  return held.contains(best) ? best + '-hold' : best;
}

@pragma('vm:entry-point')
void main() {
  assert(classifyTrafficPhasePattern([]) == 'no-cycle');
  assert(classifyTrafficPhasePattern(['green', 'green', 'red']) == 'green-hold');
  assert(classifyTrafficPhasePattern(['yellow']) == 'isolated-yellow');
  print('All tests passed!');
}