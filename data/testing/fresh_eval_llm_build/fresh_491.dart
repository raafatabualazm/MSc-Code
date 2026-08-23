@pragma('vm:entry-point')
String? mostRepeatedQrModuleBand(List<String> rows) {
  final counts = <String, int>{};
  for (final row in rows) {
    int i = 0;
    while (i < row.length) {
      int j = i + 1;
      while (j < row.length && row[j] == row[i]) {
        j++;
      }
      int len = j - i;
      if (len >= 2) {
        String key = '${row[i]}$len';
        counts[key] = (counts[key] ?? 0) + 1;
      }
      i = j;
    }
  }
  String? best;
  int bestCount = 0;
  int bestLen = -1;
  for (final entry in counts.entries) {
    int len = int.parse(entry.key.substring(1));
    if (entry.value > bestCount || (entry.value == bestCount && (len > bestLen || (len == bestLen && (best == null || entry.key.compareTo(best) < 0))))) {
      best = entry.key;
      bestCount = entry.value;
      bestLen = len;
    }
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(mostRepeatedQrModuleBand([]) == null);
  assert(mostRepeatedQrModuleBand(['##..', '..##']) == '#2');
  assert(mostRepeatedQrModuleBand(['###.', '.###', '....']) == '#3');
  print('All tests passed!');
}