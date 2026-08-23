@pragma('vm:entry-point')
int findLongestEnclosedEmptyRun(String chart) {
  int maxLen = 0;
  int i = 0;
  while (i < chart.length) {
    while (i < chart.length && chart[i] != '#') i++;
    if (i >= chart.length) break;
    int j = i + 1, run = 0;
    bool foundEnd = false;
    while (j < chart.length) {
      if (chart[j] == '.') { run++; j++; }
      else if (chart[j] == '#') { foundEnd = true; if (run > maxLen) maxLen = run; i = j; break; }
      else if (chart[j] == '|') { i = j + 1; break; }
      else { j++; }
    }
    if (!foundEnd) break;
  }
  return maxLen;
}

@pragma('vm:entry-point')
void main() {
  assert(findLongestEnclosedEmptyRun('') == 0);
  assert(findLongestEnclosedEmptyRun('#.#') == 1);
  assert(findLongestEnclosedEmptyRun('#..#') == 2);
  print('All tests passed!');
}