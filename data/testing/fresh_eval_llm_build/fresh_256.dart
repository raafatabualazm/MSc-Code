@pragma('vm:entry-point')
bool respectsDnaRankingOrder(List<String> strands) {
  int gcCount(String s) {
    int count = 0;
    for (int i = 0; i < s.length; i++) {
      if (s[i] == 'G' || s[i] == 'C') count++;
    }
    return count;
  }

  int transitions(String s) {
    int t = 0;
    for (int i = 1; i < s.length; i++) {
      if (s[i] != s[i - 1]) t++;
    }
    return t;
  }

  for (int i = 0; i < strands.length; i++) {
    for (int k = 0; k < strands[i].length; k++) {
      String ch = strands[i][k];
      if (ch != 'A' && ch != 'C' && ch != 'G' && ch != 'T') return false;
    }
    if (i == 0) continue;
    String a = strands[i - 1];
    String b = strands[i];
    int cmp = gcCount(b) - gcCount(a);
    if (cmp == 0) cmp = transitions(a) - transitions(b);
    if (cmp == 0) cmp = a.length - b.length;
    if (cmp == 0) cmp = a.compareTo(b);
    if (cmp >= 0) return false;
    int shared = 0;
    int limit = a.length < b.length ? a.length : b.length;
    for (int j = 0; j < limit; j++) {
      if (a[j] != b[j]) break;
      shared++;
    }
    if (shared == limit && gcCount(a) == gcCount(b)) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(respectsDnaRankingOrder([]) == true);
  assert(respectsDnaRankingOrder(['GG', 'CC', 'AT']) == false);
  assert(respectsDnaRankingOrder(['GC', 'GGA', 'ATA']) == true);
  print('All tests passed!');
}