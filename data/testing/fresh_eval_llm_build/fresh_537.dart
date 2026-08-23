@pragma('vm:entry-point')
bool hasStableDnaDominanceGroups(List<String> strands, int maxSpreadGap) {
  Map<String, List<Map<String, int>>> groups = {};
  for (String strand in strands) {
    if (strand.isEmpty) continue;
    Map<String, int> counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0};
    for (int i = 0; i < strand.length; i++) {
      String ch = strand[i];
      if (!counts.containsKey(ch)) return false;
      counts[ch] = counts[ch]! + 1;
    }
    String dominant = '';
    int top = 0;
    bool tied = false;
    int minSeen = strand.length;
    for (String base in counts.keys) {
      int v = counts[base]!;
      if (v > 0 && v < minSeen) minSeen = v;
      if (v > top) {
        dominant = base;
        top = v;
        tied = false;
      } else if (v == top && v > 0) {
        tied = true;
      }
    }
    if (tied || top == 0) return false;
    groups.putIfAbsent(dominant, () => []).add({'spread': top - minSeen, 'parity': strand.length % 2});
  }
  for (List<Map<String, int>> entries in groups.values) {
    for (int i = 0; i < entries.length; i++) {
      for (int j = i + 1; j < entries.length; j++) {
        int gap = (entries[i]['spread']! - entries[j]['spread']!).abs();
        if (gap > maxSpreadGap) return false;
        if (gap == 0 && entries[i]['parity'] != entries[j]['parity']) return false;
      }
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(hasStableDnaDominanceGroups(['AAAC', 'AAAACC'], 0) == true);
  assert(hasStableDnaDominanceGroups(['AAAC', 'AAACT'], 1) == false);
  assert(hasStableDnaDominanceGroups(['AACCG'], 2) == false);
  print('All tests passed!');
}