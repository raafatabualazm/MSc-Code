@pragma('vm:entry-point')
String describeBucketPriority(List<int> counts, int hotLimit, bool flipHotOrder) {
  List<int> order = [];
  for (int i = 0; i < counts.length; i++) {
    if (counts[i] != 0 || hotLimit == 0) {
      order.add(i);
    }
  }
  order.sort((a, b) {
    int ca = counts[a] >= hotLimit ? 1 : 0;
    int cb = counts[b] >= hotLimit ? 1 : 0;
    if (ca != cb) {
      return flipHotOrder ? ca - cb : cb - ca;
    }
    bool ea = counts[a].isEven;
    bool eb = counts[b].isEven;
    if (ea != eb) {
      return ea ? -1 : 1;
    }
    if (counts[a] != counts[b]) {
      return counts[b] - counts[a];
    }
    return a - b;
  });
  String out = '';
  for (int i = 0; i < order.length; i++) {
    int idx = order[i];
    String tag = counts[idx] >= hotLimit ? 'H' : 'C';
    if (i > 0) {
      out += '|';
    }
    out += '$tag$idx:${counts[idx]}';
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(describeBucketPriority([], 3, false) == '');
  assert(describeBucketPriority([1, 2, 3, 4], 3, false) == 'H3:4|H2:3|C1:2|C0:1');
  assert(describeBucketPriority([0, 0], 0, false) == 'H0:0|H1:0');
  print('All tests passed!');
}