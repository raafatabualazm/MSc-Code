@pragma('vm:entry-point')
int asciiRowReorderPenalty(List<String> rows, int minInk) {
  if (rows.isEmpty) return 0;
  List<String> kept = [];
  for (final row in rows) {
    int ink = 0;
    for (int i = 0; i < row.length; i++) {
      String c = row[i];
      if (c != ' ') ink++;
      if (c == '#' || c == '/' || c == '\\') ink++;
    }
    if (ink >= minInk || (row.isEmpty && minInk <= 0)) {
      kept.add(row);
    }
  }
  if (kept.length < 2) return kept.length;
  int density(String s) {
    int d = 0;
    for (int i = 0; i < s.length; i++) {
      if (s[i] != ' ') d++;
    }
    return d;
  }

  int jagged(String s) {
    int j = 0;
    for (int i = 1; i < s.length; i++) {
      if (s[i] != s[i - 1]) j++;
    }
    return j;
  }

  List<String> sorted = List<String>.from(kept);
  sorted.sort((a, b) {
    int byDensity = density(b) - density(a);
    if (byDensity != 0) return byDensity;
    int byJagged = jagged(a) - jagged(b);
    if (byJagged != 0) return byJagged;
    return a.compareTo(b);
  });
  int penalty = 0;
  List<bool> used = List<bool>.filled(sorted.length, false);
  for (int i = 0; i < kept.length; i++) {
    for (int j = 0; j < sorted.length; j++) {
      if (used[j] || kept[i] != sorted[j]) continue;
      penalty += (i - j).abs();
      if (kept[i].startsWith('|') && j < i) penalty++;
      used[j] = true;
      break;
    }
  }
  return penalty;
}

@pragma('vm:entry-point')
void main() {
  assert(asciiRowReorderPenalty([], 1) == 0);
  assert(asciiRowReorderPenalty(['##'], 3) == 1);
  assert(asciiRowReorderPenalty(['aa', '|##'], 1) == 3);
  print('All tests passed!');
}