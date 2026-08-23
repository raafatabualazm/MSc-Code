@pragma('vm:entry-point')
String rankDnaBaseStrings(List<String> strands, int repeatPenalty, bool descending) {
  if (strands.isEmpty) return '';
  Map<String, int> scores = {};
  for (String s in strands) {
    int score = 0;
    for (int i = 0; i < s.length; i++) {
      String ch = s[i];
      score += (ch == 'G' || ch == 'C') ? 2 : (ch == 'A' || ch == 'T') ? 1 : -3;
      if (i > 0 && s[i - 1] == ch) {
        score -= repeatPenalty;
      } else if (i > 0) {
        String pair = s.substring(i - 1, i + 1);
        if (pair == 'CG' || pair == 'GC') score += 2;
      }
      for (int j = 0; j < i; j++) {
        if (s[j] == ch && i - j > 2) {
          score++;
          break;
        }
      }
    }
    if (score < 0 && s.length > 1) score = -score ~/ 2;
    scores[s] = score;
  }
  List<String> ordered = strands.toList();
  ordered.sort((a, b) {
    int cmp = scores[a]!.compareTo(scores[b]!);
    if (descending) cmp = -cmp;
    if (cmp != 0) return cmp;
    cmp = b.length.compareTo(a.length);
    return cmp != 0 ? cmp : a.compareTo(b);
  });
  return ordered.join(':');
}

@pragma('vm:entry-point')
void main() {
  assert(rankDnaBaseStrings(['AT', 'GC'], 1, false) == 'AT:GC');
  assert(rankDnaBaseStrings([], 2, true) == '');
  assert(rankDnaBaseStrings(['GG', 'GC'], 2, false) == 'GG:GC');
  print('All tests passed!');
}