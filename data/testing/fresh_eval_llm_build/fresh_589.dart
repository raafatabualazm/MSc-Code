@pragma('vm:entry-point')
int packetSizeStateScore(String tape) {
  int score = 0;
  int size = 0;
  bool reading = false;
  bool jumbo = false;
  bool invert = false;
  for (int i = 0; i <= tape.length; i++) {
    String ch = i < tape.length ? tape[i] : ';';
    if (ch == '#') {
      while (i + 1 < tape.length && tape[i + 1] != ';') {
        i++;
      }
      continue;
    }
    if (ch == '!') {
      invert = !invert;
      continue;
    }
    int code = ch.codeUnitAt(0);
    if (code >= 48 && code <= 57) {
      size = reading ? size * 10 + code - 48 : code - 48;
      reading = true;
      continue;
    }
    if (ch == 'J' && reading) {
      jumbo = true;
      continue;
    }
    if (ch != ';' && ch != ',') {
      return -99;
    }
    if (reading) {
      int value = jumbo ? size * 2 : size;
      if (value >= 64 && value <= 1500) {
        score += invert ? -2 : 2;
      } else if (value < 64) {
        score += invert ? 1 : -1;
      } else {
        score += invert ? -3 : 3;
      }
    }
    size = 0;
    reading = false;
    jumbo = false;
    invert = false;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(packetSizeStateScore('64;') == 2);
  assert(packetSizeStateScore('63;') == -1);
  assert(packetSizeStateScore('64,10,2000;') == 4);
  print('All tests passed!');
}