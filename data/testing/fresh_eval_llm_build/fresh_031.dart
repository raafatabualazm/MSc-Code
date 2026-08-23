@pragma('vm:entry-point')
int passwordAuditOrderingScore(List<String> passwords) {
  int violations(String s) {
    bool lower = false, upper = false, digit = false;
    for (final c in s.codeUnits) {
      if (c >= 97 && c <= 122) lower = true;
      else if (c >= 65 && c <= 90) upper = true;
      else if (c >= 48 && c <= 57) digit = true;
    }
    return (lower ? 0 : 1) + (upper ? 0 : 1) + (digit ? 0 : 1) + (s.length >= 6 ? 0 : 1);
  }

  final sorted = List<String>.from(passwords);
  sorted.sort((a, b) {
    final va = violations(a), vb = violations(b);
    if (va != vb) return va - vb;
    if (a.length != b.length) return b.length - a.length;
    return a.compareTo(b);
  });

  int score = 0;
  for (int i = 1; i < sorted.length; i++) {
    final diff = violations(sorted[i]) - violations(sorted[i - 1]);
    if (diff == 0) {
      score += (sorted[i].length - sorted[i - 1].length).abs() <= 1 ? 2 : 1;
    } else if (diff == 1) {
      score -= 1;
    } else {
      score -= 2;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(passwordAuditOrderingScore([]) == 0);
  assert(passwordAuditOrderingScore(['ABC123', 'abc123']) == 2);
  assert(passwordAuditOrderingScore(['aB3def', '123456', '']) == -4);
  print('All tests passed!');
}