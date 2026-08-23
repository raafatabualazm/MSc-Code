@pragma('vm:entry-point')
List<String> scanSatellitePassWindows(String script) {
  List<String> out = [];
  int i = 0;
  while (i < script.length) {
    while (i < script.length && (script[i] == ' ' || script[i] == ';')) {
      i++;
    }
    if (i >= script.length) return out;
    String id = '';
    while (i < script.length && script.codeUnitAt(i) >= 65 && script.codeUnitAt(i) <= 90) {
      id += script[i++];
    }
    if (id.isEmpty || i >= script.length || (script[i] != '+' && script[i] != '-')) {
      while (i < script.length && script[i] != ';') {
        i++;
      }
      continue;
    }
    bool open = script[i++] == '+';
    int value = 0;
    bool sawDigit = false;
    while (i < script.length && script.codeUnitAt(i) >= 48 && script.codeUnitAt(i) <= 57) {
      sawDigit = true;
      value = value * 10 + script.codeUnitAt(i++) - 48;
    }
    if (!sawDigit || i >= script.length) return out;
    String band = script[i++];
    if (!open || (band != 'L' && band != 'H')) continue;
    if ((band == 'L' && value >= 3) || (band == 'H' && value >= 6)) {
      out.add(id + ':' + value.toString() + band);
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(scanSatellitePassWindows("SAT+3L").toString() == "[SAT:3L]");
  assert(scanSatellitePassWindows("SAT-9H;ORB+6H").toString() == "[ORB:6H]");
  assert(scanSatellitePassWindows("BAD;ZEN+2L").isEmpty);
  print('All tests passed!');
}