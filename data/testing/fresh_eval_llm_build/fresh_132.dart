@pragma('vm:entry-point')
List<int> scanSpellEditTokens(String log) {
  var counts = [0, 0, 0, 0];
  String pending = '';
  for (var i = 0; i < log.length; i++) {
    var c = log[i];
    if (c == '+' || c == '-' || c == '~') {
      if (pending.isNotEmpty) counts[3]++;
      pending = c;
    } else if (pending.isNotEmpty && c.codeUnitAt(0) >= 97 && c.codeUnitAt(0) <= 122) {
      counts[pending == '+' ? 0 : pending == '-' ? 1 : 2]++;
      pending = '';
    }
  }
  if (pending.isNotEmpty) counts[3]++;
  return counts;
}

@pragma('vm:entry-point')
void main() {
  assert(scanSpellEditTokens("+a-b~c").toString() == '[1, 1, 1, 0]');
  assert(scanSpellEditTokens("++a").toString() == '[1, 0, 0, 1]');
  assert(scanSpellEditTokens("word").toString() == '[0, 0, 0, 0]');
  print('All tests passed!');
}