@pragma('vm:entry-point')
List<String> extractQualifiedCandidates(String voteLog, int minVotes) {
  if (voteLog.trim().isEmpty) return [];
  final Map<String, int> totals = {};
  final entries = voteLog.split(';');
  for (final entry in entries) {
    final trimmed = entry.trim();
    if (trimmed.isEmpty) continue;
    final colonIdx = trimmed.indexOf(':');
    if (colonIdx <= 0) continue;
    final name = trimmed.substring(0, colonIdx).trim();
    if (name.isEmpty) continue;
    final votePart = trimmed.substring(colonIdx + 1).trim();
    if (votePart.isEmpty) continue;
    bool isNumeric = true;
    for (int i = 0; i < votePart.length; i++) {
      final c = votePart.codeUnitAt(i);
      if (c < 48 || c > 57) { isNumeric = false; break; }
    }
    if (!isNumeric) continue;
    final votes = int.parse(votePart);
    totals[name] = (totals[name] ?? 0) + votes;
  }
  final qualified = <String>[];
  for (final entry in totals.entries) {
    if (entry.value >= minVotes) {
      qualified.add('${entry.key}(${entry.value})');
    }
  }
  qualified.sort((a, b) {
    final aName = a.substring(0, a.indexOf('('));
    final bName = b.substring(0, b.indexOf('('));
    final aVal = int.parse(a.substring(a.indexOf('(') + 1, a.length - 1));
    final bVal = int.parse(b.substring(b.indexOf('(') + 1, b.length - 1));
    if (bVal != aVal) return bVal.compareTo(aVal);
    return aName.compareTo(bName);
  });
  return qualified;
}

@pragma('vm:entry-point')
void main() {
  assert(extractQualifiedCandidates('Alice:30;Bob:20;Alice:15;Bob:5;Carol:10', 25).toString() == '[Alice(45), Bob(25)]');
  assert(extractQualifiedCandidates('', 1).length == 0);
  assert(extractQualifiedCandidates('Zara:10;Zara:10', 20).toString() == '[Zara(20)]');
  print('All tests passed!');
}