@pragma('vm:entry-point')
List<int> auditManifestBursts(String manifest) {
  int loaded = 0, warnings = 0, checksum = 0;
  String current = '';
  int count = 0;
  bool sawDigit = false;
  for (int i = 0; i <= manifest.length; i++) {
    String ch = i < manifest.length ? manifest[i] : '#';
    bool isDigit = i < manifest.length && ch.codeUnitAt(0) >= 48 && ch.codeUnitAt(0) <= 57;
    if (isDigit) {
      count = count * 10 + int.parse(ch);
      sawDigit = true;
    } else {
      if (current.isNotEmpty) {
        int units = sawDigit ? count : 1;
        loaded += units;
        checksum = (checksum + (current.codeUnitAt(0) - 64) * units) % 97;
        if (units == 0) {
          warnings += 5;
        } else if (units > 9) {
          warnings += 2;
        }
        if (i < manifest.length && ch == current) warnings += 3;
      }
      current = i < manifest.length ? ch : '';
      count = 0;
      sawDigit = false;
    }
  }
  return [loaded, warnings, checksum];
}

@pragma('vm:entry-point')
void main() {
  assert(auditManifestBursts('').toString() == '[0, 0, 0]');
  assert(auditManifestBursts('AA').toString() == '[2, 3, 2]');
  assert(auditManifestBursts('A10').toString() == '[10, 2, 10]');
  print('All tests passed!');
}