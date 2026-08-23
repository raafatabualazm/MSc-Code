@pragma('vm:entry-point')
int tallyWifiBinTransitions(List<String> events, int weakMax, int strongMin) {
  List<int> bins = [];
  for (String e in events) {
    if (e == 'DROP') {
      if (bins.isNotEmpty) bins.removeLast();
    } else if (e == 'REPEAT') {
      if (bins.isNotEmpty) bins.add(bins.last);
    } else {
      int v = int.parse(e);
      if (v <= weakMax) {
        bins.add(0);
      } else if (v >= strongMin) {
        bins.add(2);
      } else {
        bins.add(1);
      }
    }
  }
  int score = 0;
  for (int i = 0; i < bins.length; i++) {
    score += bins[i] == 2 ? 3 : (bins[i] == 1 ? 1 : -2);
    if (i > 0 && bins[i] != bins[i - 1]) score++;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(tallyWifiBinTransitions([], -70, -40) == 0);
  assert(tallyWifiBinTransitions(['-80', 'REPEAT', '-39'], -70, -40) == 0);
  assert(tallyWifiBinTransitions(['-50', 'REPEAT', 'DROP'], -70, -40) == 1);
  print('All tests passed!');
}