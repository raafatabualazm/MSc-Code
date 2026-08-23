@pragma('vm:entry-point')
String findFirstLogAtSeverity(List<String> logLines, String minSeverity) {
  final Map<String, int> rank = {
    'DEBUG': 0, 'INFO': 1, 'WARN': 2, 'ERROR': 3, 'FATAL': 4
  };
  final int minRank = rank[minSeverity] ?? 0;
  String result = 'none';
  int bestTs = -1;
  for (final line in logLines) {
    final parts = line.split(' ');
    if (parts.length < 3) continue;
    final int? ts = int.tryParse(parts[0]);
    final int? sev = rank[parts[1]];
    if (ts == null || sev == null) continue;
    if (sev >= minRank) {
      if (bestTs == -1 || ts < bestTs) {
        bestTs = ts;
        result = parts[0];
      }
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(findFirstLogAtSeverity(['100 DEBUG start', '200 INFO req', '300 WARN timeout', '400 ERROR crash', '500 FATAL oom'], 'WARN') == '300');
  assert(findFirstLogAtSeverity([], 'ERROR') == 'none');
  assert(findFirstLogAtSeverity(['50 INFO msg'], 'ERROR') == 'none');
  print('All tests passed!');
}