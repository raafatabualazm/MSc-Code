@pragma('vm:entry-point')
String logWindowGapSummary(String logs) {
  if (logs.trim().isEmpty) return 'empty';
  final Map<int, Set<int>> days = {};
  for (final part in logs.split(';')) {
    if (part.isEmpty || !part.contains('@') || !part.contains('+')) continue;
    final left = part.split('@');
    final right = left[1].split('+');
    final day = int.tryParse(left[0]);
    final start = int.tryParse(right[0]);
    final len = int.tryParse(right[1]);
    if (day == null || start == null || len == null || len <= 0) continue;
    days.putIfAbsent(day, () => <int>{});
    for (int m = start; m < start + len; m++) {
      days[day]!.add(m);
    }
  }
  if (days.isEmpty) return 'empty';
  final keys = days.keys.toList()..sort();
  final out = <String>[];
  for (final day in keys) {
    final used = days[day]!.toList()..sort();
    int longest = 0;
    for (int i = 1; i < used.length; i++) {
      final gap = used[i] - used[i - 1] - 1;
      if (gap > longest) longest = gap;
      if (gap > 7) {
        longest = 9;
        break;
      }
    }
    if (longest == 0) {
      out.add('$day=solid');
    } else if (longest == 1) {
      out.add('$day=tiny');
    } else {
      out.add('$day=gap$longest');
    }
  }
  return out.join(',');
}

@pragma('vm:entry-point')
void main() {
  assert(logWindowGapSummary('1@0+3') == '1=solid');
  assert(logWindowGapSummary('1@0+1;1@2+1') == '1=tiny');
  assert(logWindowGapSummary('1@0+1;1@10+1') == '1=gap9');
  print('All tests passed!');
}