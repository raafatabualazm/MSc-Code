@pragma('vm:entry-point')
int tallyDiceCipherRounds(String rounds) {
  int score = 0;
  int count = 0;
  for (int i = 0; i < rounds.length; i++) {
    int code = rounds.codeUnitAt(i);
    if (code >= 48 && code <= 57) {
      count = count * 10 + code - 48;
    } else {
      int repeat = count == 0 ? 1 : count;
      int face = code & 31;
      if (code >= 65 && code <= 90) {
        score += face.isEven ? face * repeat : repeat;
      } else {
        score -= face >= 4 ? face + repeat : repeat;
      }
      if (repeat > 2 && face == 6) score += code >= 65 && code <= 90 ? 2 : -2;
      count = 0;
    }
  }
  return count > 0 ? score - count : score;
}

@pragma('vm:entry-point')
void main() {
  assert(tallyDiceCipherRounds('3F') == 20);
  assert(tallyDiceCipherRounds('2dE') == -5);
  assert(tallyDiceCipherRounds('2B4') == 0);
  print('All tests passed!');
}