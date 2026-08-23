@pragma('vm:entry-point')
List<String> arrangeCentRoundings(List<int> centsValues) {
  List<Map<String, Object>> rows = [];
  for (int cents in centsValues) {
    int rounded = cents >= 0 ? ((cents + 2) ~/ 5) * 5 : ((cents - 2) ~/ 5) * 5;
    int diff = rounded - cents;
    String tag;
    if (diff > 0) {
      tag = 'UP';
    } else if (diff < 0) {
      tag = 'DOWN';
    } else {
      tag = 'EVEN';
    }
    rows.add({'label': '$tag:${diff.abs()}@$cents', 'gap': diff.abs(), 'cents': cents, 'tag': tag});
  }
  rows.sort((a, b) {
    int byGap = (b['gap'] as int).compareTo(a['gap'] as int);
    if (byGap != 0) return byGap;
    if (a['tag'] != b['tag']) return (a['tag'] as String).compareTo(b['tag'] as String);
    return (a['cents'] as int).compareTo(b['cents'] as int);
  });
  return rows.map((e) => e['label'] as String).toList();
}

@pragma('vm:entry-point')
void main() {
  assert(arrangeCentRoundings([]).toString() == '[]');
  assert(arrangeCentRoundings([100]).toString() == '[EVEN:0@100]');
  assert(arrangeCentRoundings([102, 103]).toString() == '[DOWN:2@102, UP:2@103]');
  print('All tests passed!');
}