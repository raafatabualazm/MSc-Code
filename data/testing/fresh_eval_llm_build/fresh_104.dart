@pragma('vm:entry-point')
List<String> classifyTrafficPhasePatterns(List<String> phases) {
  if (phases.isEmpty) return [];
  final counts = <String, int>{};
  final order = <String>[];
  for (final p in phases) {
    if (p.isEmpty) continue;
    counts[p] = (counts[p] ?? 0) + 1;
    if (!order.contains(p)) {
      order.add(p);
    }
  }
  if (counts.isEmpty) return [];
  final result = <String>[];
  for (final phase in order) {
    int longest = 0;
    for (int i = 0; i < phases.length; i++) {
      if (phases[i] != phase) continue;
      int run = 1;
      for (int j = i + 1; j < phases.length && phases[j] == phase; j++) {
        run++;
      }
      if (run > longest) longest = run;
    }
    final total = counts[phase]!;
    if (total == 1) {
      result.add('$phase:solo');
    } else if (longest >= 3 && total > longest) {
      result.add('$phase:surge');
    } else if (longest >= 3) {
      result.add('$phase:hold');
    } else if (total - longest >= 2) {
      result.add('$phase:scatter');
    } else {
      result.add('$phase:steady');
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(classifyTrafficPhasePatterns([]).length == 0);
  assert(classifyTrafficPhasePatterns(['G','G','G']).toString() == '[G:hold]');
  assert(classifyTrafficPhasePatterns(['R','R','R','G','R']).toString() == '[R:surge, G:solo]');
  print('All tests passed!');
}