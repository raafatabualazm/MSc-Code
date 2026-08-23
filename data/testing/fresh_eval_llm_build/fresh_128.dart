@pragma('vm:entry-point')
List<String> arrangeTideHeightBands(List<String> readings) {
  var items = <Map<String, Object>>[];
  for (var entry in readings) {
    var parts = entry.split('@');
    var level = double.parse(parts[1]);
    var tag = 'calm';
    var rank = 2;
    if (level < 0.0) {
      tag = 'retreat'; rank = 3;
    } else if (level >= 4.0) {
      tag = 'surge'; rank = 0;
    } else if (level >= 2.0) {
      tag = 'watch'; rank = 1;
    }
    items.add({'text': '${parts[0]}:$tag', 'rank': rank, 'level': level});
  }
  items.sort((a, b) {
    var byRank = (a['rank'] as int).compareTo(b['rank'] as int);
    if (byRank != 0) return byRank;
    var byLevel = (b['level'] as double).compareTo(a['level'] as double);
    return byLevel != 0 ? byLevel : (a['text'] as String).compareTo(b['text'] as String);
  });
  return [for (var item in items) item['text'] as String];
}

@pragma('vm:entry-point')
void main() {
  assert(arrangeTideHeightBands([]).toString() == [].toString());
  assert(arrangeTideHeightBands(['North@4.0']).toString() == ['North:surge'].toString());
  assert(arrangeTideHeightBands(['A@2.0','B@1.5']).first == 'A:watch');
  print('All tests passed!');
}