@pragma('vm:entry-point')
List<int> orderDnaIndicesBySignal(List<String> strands) {
  List<int> scores = List.filled(strands.length, 0);
  for (int i = 0; i < strands.length; i++) {
    String s = strands[i];
    int score = 0;
    for (int j = 0; j < s.length; j++) {
      String c = s[j];
      if (c == 'G' || c == 'C') {
        score += 2;
        if (j > 0 && s[j - 1] == c) score += 1;
      } else if (c == 'A' || c == 'T') {
        score += 1;
        if (j > 0 && s[j - 1] == c) score -= 1;
      } else {
        score -= 3;
      }
    }
    scores[i] = score;
  }
  List<int> order = List<int>.generate(strands.length, (i) => i);
  order.sort((a, b) {
    if (scores[a] != scores[b]) return scores[b] - scores[a];
    if (strands[a].length != strands[b].length) {
      return strands[a].length - strands[b].length;
    }
    return a - b;
  });
  return order;
}

@pragma('vm:entry-point')
void main() {
  assert(orderDnaIndicesBySignal(['GC', 'AT']).toString() == '[0, 1]');
  assert(orderDnaIndicesBySignal([]).length == 0);
  assert(orderDnaIndicesBySignal(['AAA', 'GG']).toString() == '[1, 0]');
  print('All tests passed!');
}