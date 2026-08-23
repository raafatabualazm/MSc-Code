@pragma('vm:entry-point')
bool verifySatellitePassWindows(List<String> windows) {
  final parsed = <Map<String, Object>>[];
  for (final entry in windows) {
    final parts = entry.split('|');
    if (parts.length != 4) return false;
    final start = int.tryParse(parts[1]);
    final end = int.tryParse(parts[2]);
    final priority = int.tryParse(parts[3]);
    if (parts[0].isEmpty || start == null || end == null || priority == null) {
      return false;
    }
    if (start < 0 || end <= start || priority < 0) return false;
    parsed.add({'sat': parts[0], 'start': start, 'end': end, 'priority': priority});
  }
  parsed.sort((a, b) {
    final s = (a['start'] as int).compareTo(b['start'] as int);
    if (s != 0) return s;
    final p = (b['priority'] as int).compareTo(a['priority'] as int);
    if (p != 0) return p;
    return (a['sat'] as String).compareTo(b['sat'] as String);
  });
  for (int i = 0; i < parsed.length; i++) {
    int overlaps = 1;
    for (int j = i + 1; j < parsed.length; j++) {
      if ((parsed[j]['start'] as int) >= (parsed[i]['end'] as int)) break;
      overlaps++;
      if (overlaps > 2) return false;
      if (parsed[i]['sat'] == parsed[j]['sat']) return false;
      if (parsed[i]['priority'] == parsed[j]['priority']) return false;
      if ((parsed[i]['start'] as int) == (parsed[j]['start'] as int) &&
          (parsed[i]['end'] as int) == (parsed[j]['end'] as int)) {
        return false;
      }
    }
    if (i > 0 && (parsed[i]['start'] as int) < (parsed[i - 1]['end'] as int)) {
      if ((parsed[i]['priority'] as int) < (parsed[i - 1]['priority'] as int)) {
        return false;
      }
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(verifySatellitePassWindows([]) == true);
  assert(verifySatellitePassWindows(['A|0|3|2','B|2|5|3']) == true);
  assert(verifySatellitePassWindows(['A|0|4|1','B|1|5|0']) == false);
  print('All tests passed!');
}