@pragma('vm:entry-point')
int measureShelfCodeDisarray(List<String> codes) {
  if (codes.isEmpty) return 0;
  List<String> valid = [];
  for (String code in codes) {
    int split = 0;
    while (split < code.length) {
      int unit = code.codeUnitAt(split);
      if (unit < 65 || unit > 90) break;
      split++;
    }
    if (split == 0 || split == code.length) continue;
    valid.add(code);
  }
  if (valid.length < 2) return 0;
  List<String> ordered = List<String>.from(valid);
  ordered.sort((a, b) {
    int sa = 0, sb = 0;
    while (sa < a.length && a.codeUnitAt(sa) >= 65 && a.codeUnitAt(sa) <= 90) { sa++; }
    while (sb < b.length && b.codeUnitAt(sb) >= 65 && b.codeUnitAt(sb) <= 90) { sb++; }
    String pa = a.substring(0, sa), pb = b.substring(0, sb);
    if (pa != pb) return pa.compareTo(pb);
    int na = int.parse(a.substring(sa)), nb = int.parse(b.substring(sb));
    if (na != nb) return nb.compareTo(na);
    return a.length.compareTo(b.length);
  });
  int score = 0;
  for (int i = 0; i < valid.length; i++) {
    if (valid[i] == ordered[i]) continue;
    int found = -1;
    for (int j = 0; j < valid.length; j++) {
      if (valid[j] == ordered[i]) { found = j; break; }
    }
    if (found == -1) return -1;
    score += (found - i).abs();
    if (ordered[i][0] != valid[i][0]) {
      score += 2;
    } else if (ordered[i].length != valid[i].length) {
      score += 1;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(measureShelfCodeDisarray([]) == 0);
  assert(measureShelfCodeDisarray(['A1', 'A2']) == 2);
  assert(measureShelfCodeDisarray(['B1', 'A9']) == 6);
  print('All tests passed!');
}