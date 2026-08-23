@pragma('vm:entry-point')
int electionAuditScore(List<String> tallies) {
  if (tallies.isEmpty) return 0;
  Map<String, Map<String, int>> districts = {};
  for (var row in tallies) {
    var parts = row.split('|');
    if (parts.length != 3) continue;
    var votes = int.tryParse(parts[2]);
    if (votes == null || votes < 0) continue;
    districts.putIfAbsent(parts[0], () => {});
    var bucket = districts[parts[0]]!;
    bucket[parts[1]] = (bucket[parts[1]] ?? 0) + votes;
  }
  int score = 0;
  for (var bucket in districts.values) {
    if (bucket.isEmpty) continue;
    int top = -1, second = -1, leaders = 0, total = 0;
    bool hasZero = false;
    for (var votes in bucket.values) {
      total += votes;
      if (votes == 0) hasZero = true;
      if (votes > top) {
        second = top;
        top = votes;
        leaders = 1;
      } else if (votes == top) {
        leaders++;
      } else if (votes > second) {
        second = votes;
      }
    }
    if (bucket.length < 2) {
      score += total % 7;
    } else if (leaders > 1) {
      score += 11;
    } else if (top - second == 1) {
      score += 5;
    } else if (second >= 0 && top > second * 2) {
      score -= 3;
    } else {
      score += 2;
    }
    if (hasZero) score += 1;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(electionAuditScore([]) == 0);
  assert(electionAuditScore(['north|A|10', 'north|B|9']) == 5);
  assert(electionAuditScore(['n|A|0', 'n|B|4']) == -2);
  print('All tests passed!');
}