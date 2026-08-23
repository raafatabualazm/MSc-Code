@pragma('vm:entry-point')
int auditShelfCodePenalty(String records) {
  int score = 0;
  Map<String, int> lastAisle = {};
  for (String raw in records.split(',')) {
    String token = raw.trim();
    if (token.isEmpty) continue;
    bool flagged = token.endsWith('!');
    String core = flagged ? token.substring(0, token.length - 1) : token;
    if (core.length < 3) {
      score += 4;
      continue;
    }
    int section = core.codeUnitAt(0) - 64;
    if (section < 1 || section > 26) {
      score += 4;
      continue;
    }
    int digitSum = 0, aisle = 0;
    bool bad = false;
    for (int i = 1; i < core.length; i++) {
      int c = core.codeUnitAt(i);
      if (c < 48 || c > 57) {
        bad = true;
        break;
      }
      int d = c - 48;
      digitSum += d;
      aisle = aisle * 10 + d;
      if (i > 1 && core.codeUnitAt(i - 1) == c) score++;
    }
    if (bad) {
      score += 4;
      continue;
    }
    String key = core[0];
    if (lastAisle.containsKey(key) && aisle < lastAisle[key]!) score += 5;
    lastAisle[key] = aisle;
    score += flagged ? (digitSum.isEven ? 3 : -1) : (digitSum % section == 0 ? 2 : -2);
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(auditShelfCodePenalty('A12') == 2);
  assert(auditShelfCodePenalty('B12') == -2);
  assert(auditShelfCodePenalty('A21,A19') == 9);
  print('All tests passed!');
}