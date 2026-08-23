@pragma('vm:entry-point')
int countMisroutedManifestEntries(String manifest) {
  if (manifest.isEmpty) return 0;
  final entries = manifest.split(';');
  int misrouted = 0;
  for (final entry in entries) {
    if (entry.trim().isEmpty) continue;
    final parts = entry.split(':');
    if (parts.length != 3) {
      misrouted++;
      continue;
    }
    final portStr = parts[0].trim();
    final weightStr = parts[1].trim();
    final flag = parts[2].trim();
    if (portStr.isEmpty) {
      misrouted++;
      continue;
    }
    final portChar = portStr[0].toUpperCase().codeUnitAt(0);
    if (portChar < 65 || portChar > 90) {
      misrouted++;
      continue;
    }
    final weight = int.tryParse(weightStr);
    if (weight == null) {
      misrouted++;
      continue;
    }
    final maxWeight = (portChar <= 77) ? 50 : 30;
    bool overweightHazard = weight > maxWeight && flag == 'H';
    bool suspiciousPriority = weight < 5 && flag == 'P';
    if (overweightHazard || suspiciousPriority) misrouted++;
  }
  return misrouted;
}

@pragma('vm:entry-point')
void main() {
  assert(countMisroutedManifestEntries('') == 0);
  assert(countMisroutedManifestEntries('A:55:H;N:35:H;B:3:P') == 3);
  assert(countMisroutedManifestEntries('A:abc:H;B:3:X;Z:25:H') == 1);
  print('All tests passed!');
}