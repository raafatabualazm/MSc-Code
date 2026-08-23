@pragma('vm:entry-point')
String auditManifestFlags(List<int> manifests) {
  if (manifests.isEmpty) return 'EMPTY';
  int hazard = 0;
  int cold = 0;
  int reroute = 0;
  for (final raw in manifests) {
    int code = raw & 255;
    int lowCount = 0;
    for (int b = 0; b < 4; b++) {
      if (((code >> b) & 1) == 1) lowCount++;
    }
    if ((code & 0x81) == 0x81) {
      hazard += lowCount >= 2 ? 2 : 1;
    } else if (lowCount == 0) {
      reroute++;
    } else {
      int rotated = ((code << 1) | (code >> 7)) & 255;
      if ((rotated & 0x04) != 0 && (code & 0x04) == 0) cold++;
    }
  }
  return '$hazard|$cold|$reroute';
}

@pragma('vm:entry-point')
void main() {
  assert(auditManifestFlags([]) == 'EMPTY');
  assert(auditManifestFlags([131]) == '2|0|0');
  assert(auditManifestFlags([2, 0, 130]) == '0|2|1');
  print('All tests passed!');
}