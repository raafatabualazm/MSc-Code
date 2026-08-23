@pragma('vm:entry-point')
List<int> qrModuleTransitionCounts(String script) {
  List<int> out = [];
  int transitions = 0;
  int current = -1;
  bool invert = false;
  int width = 0;
  for (int i = 0; i < script.length; i++) {
    String ch = script[i];
    if (ch == '/') {
      if (width == 0) return [];
      out.add(transitions);
      transitions = 0;
      current = -1;
      invert = false;
      width = 0;
      continue;
    }
    if (ch == '?') {
      invert = !invert;
      continue;
    }
    if (ch != 'B' && ch != 'W') return [];
    int color = (ch == 'B') ^ invert ? 1 : 0;
    int count = 0;
    while (i + 1 < script.length) {
      int d = script.codeUnitAt(i + 1) - 48;
      if (d < 0 || d > 9) break;
      count = count * 10 + d;
      i++;
    }
    if (count == 0) count = 1;
    for (int k = 0; k < count; k++) {
      if (current != -1 && current != color) transitions++;
      current = color;
      width++;
    }
  }
  if (width > 0) out.add(transitions);
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(qrModuleTransitionCounts('BWB').toString() == '[2]');
  assert(qrModuleTransitionCounts('B2W3/B').toString() == '[1, 0]');
  assert(qrModuleTransitionCounts('/').isEmpty);
  print('All tests passed!');
}