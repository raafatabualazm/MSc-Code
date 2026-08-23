@pragma('vm:entry-point')
int scoreMorseRelayOrdering(List<String> signals) {
  if (signals.isEmpty) return 0;
  List<String> sorted = List<String>.from(signals);
  sorted.sort((a, b) {
    int lenOrder = a.length.compareTo(b.length);
    if (lenOrder != 0) return lenOrder;
    int dashA = '-'.allMatches(a).length;
    int dashB = '-'.allMatches(b).length;
    if (dashA != dashB) return dashB.compareTo(dashA);
    return a.compareTo(b);
  });
  int score = 0;
  for (int i = 0; i < sorted.length; i++) {
    String current = sorted[i];
    if (current.isEmpty) continue;
    if (i > 0 && current == sorted[i - 1]) {
      score -= 2;
      continue;
    }
    int local = 0;
    for (int j = 0; j < current.length; j++) {
      local += current[j] == '-' ? 2 : 1;
      if (j > 0) local += current[j] == current[j - 1] ? 1 : -1;
    }
    if (i > 0) {
      String prev = sorted[i - 1];
      int limit = current.length < prev.length ? current.length : prev.length;
      int shared = 0;
      for (int k = 0; k < limit; k++) {
        if (current[k] != prev[k]) break;
        shared++;
      }
      if (shared == limit && current.length != prev.length) {
        score += shared * 2;
      } else if (shared == 0) {
        score -= 1;
      } else {
        score += shared;
      }
    }
    score += local;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(scoreMorseRelayOrdering([]) == 0);
  assert(scoreMorseRelayOrdering(['.', '..', '...']) == 15);
  assert(scoreMorseRelayOrdering(['.-', '.-']) == 0);
  print('All tests passed!');
}