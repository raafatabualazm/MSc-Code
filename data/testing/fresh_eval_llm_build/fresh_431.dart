@pragma('vm:entry-point')
List<int> extractManifestNetLoads(String manifest) {
  List<int> loads = [];
  for (String rawSection in manifest.split(';')) {
    String section = rawSection.trim();
    if (section.isEmpty) {
      continue;
    }
    int total = 0;
    bool invalid = false;
    for (int i = 0; i < section.length; i++) {
      String c = section[i];
      if (c == '#') {
        invalid = true;
        break;
      }
      if (c != '+' && c != '-') {
        continue;
      }
      int j = i + 1;
      while (j < section.length && section[j] == ' ') {
        j++;
      }
      int value = 0;
      bool foundDigit = false;
      while (j < section.length) {
        int code = section.codeUnitAt(j);
        if (code < 48 || code > 57) {
          break;
        }
        foundDigit = true;
        value = value * 10 + code - 48;
        j++;
      }
      if (!foundDigit) {
        continue;
      }
      if (j < section.length && section[j] == '!') {
        value *= 2;
      }
      total += c == '+' ? value : -value;
      i = j - 1;
    }
    if (!invalid && total != 0) {
      loads.add(total);
    }
  }
  return loads;
}

@pragma('vm:entry-point')
void main() {
  assert(extractManifestNetLoads('').toString() == '[]');
  assert(extractManifestNetLoads('dock+5,-2').toString() == '[3]');
  assert(extractManifestNetLoads('x+7,-7').length == 0);
  print('All tests passed!');
}