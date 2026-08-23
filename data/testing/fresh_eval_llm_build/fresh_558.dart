@pragma('vm:entry-point')
int computeRunSymmetryScore(String art) {
  int score = 0;
  for (var line in art.split('\n')) {
    if (line.isEmpty) continue;
    int i = 0;
    while (i < line.length) {
      if (line[i] != '.' && line[i] != '#') return -1;
      var ch = line[i];
      int start = i;
      while (i < line.length && line[i] == ch) i++;
      int len = i - start;
      if (ch == '#' && i < line.length && line[i] == '.') {
        start = i;
        while (i < line.length && line[i] == '.') i++;
        int dotLen = i - start;
        if (dotLen > len) score += dotLen - len;
        else if (dotLen < len) score -= len - dotLen;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(computeRunSymmetryScore('') == 0);
  assert(computeRunSymmetryScore('#..') == 1);
  assert(computeRunSymmetryScore('a') == -1);
  print('All tests passed!');
}