@pragma('vm:entry-point')
String? classifyWifiBinCascade(List<int> bins) {
  if (bins.isEmpty) return null;
  String? best;
  int bestParts = -1;
  void search(int at, String path, String? last, int parts) {
    if (at == bins.length) {
      if (parts > bestParts || (parts == bestParts && (best == null || path.compareTo(best!) < 0))) {
        bestParts = parts;
        best = path;
      }
      return;
    }
    for (int len = 1; len <= 3; len++) {
      if (at + len > bins.length) continue;
      int low = bins[at], high = bins[at], sum = 0;
      for (int i = at; i < at + len; i++) {
        if (bins[i] < low) low = bins[i];
        if (bins[i] > high) high = bins[i];
        sum += bins[i];
      }
      String? label;
      if (high < 2) label = 'W';
      else if (low > 4) label = 'H';
      else if (high - low <= 1 && sum >= len * 2 && sum <= len * 4) label = 'S';
      else continue;
      if (label == last) continue;
      search(at + len, path.isEmpty ? label : '$path>$label', label, parts + 1);
    }
  }
  search(0, '', null, 0);
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(classifyWifiBinCascade([0, 3, 5]) == 'W>S>H');
  assert(classifyWifiBinCascade([2, 3, 4]) == null);
  assert(classifyWifiBinCascade([5, 5, 0, 0]) == 'H>W');
  print('All tests passed!');
}