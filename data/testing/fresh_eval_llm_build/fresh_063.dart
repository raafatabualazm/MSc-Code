@pragma('vm:entry-point')
int countPriorityManifestLoads(String manifest, int minZone) {
  int zone = -1;
  int count = 0;
  for (final c in manifest.split('')) {
    if (c == 'C') {
      zone = -2;
    } else if (zone == -2 && c.codeUnitAt(0) >= 48 && c.codeUnitAt(0) <= 57) {
      zone = c.codeUnitAt(0) - 48;
    } else if (c == '!' && zone >= minZone) {
      count++;
      zone = -1;
    } else {
      zone = -1;
    }
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countPriorityManifestLoads('C5!', 5) == 1);
  assert(countPriorityManifestLoads('C4!C5!', 5) == 1);
  assert(countPriorityManifestLoads('C8?!', 8) == 0);
  print('All tests passed!');
}