@pragma('vm:entry-point')
int countEmptySeatPairs(String chart) {
  int countEmptyPairs = 0;
  bool prevEmpty = false;
  for (int i = 0; i < chart.length; i++) {
    String c = chart[i];
    if (c == '\n') {
      prevEmpty = false;
    } else if (c == '.') {
      if (prevEmpty) countEmptyPairs++;
      prevEmpty = true;
    } else {
      prevEmpty = false;
    }
  }
  return countEmptyPairs;
}

@pragma('vm:entry-point')
void main() {
  assert(countEmptySeatPairs('..') == 1);
  assert(countEmptySeatPairs('...') == 2);
  assert(countEmptySeatPairs('..\n..') == 2);
  print('All tests passed!');
}