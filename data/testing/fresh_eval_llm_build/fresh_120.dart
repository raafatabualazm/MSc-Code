@pragma('vm:entry-point')
int evaluateTheaterAislePressure(int seatingMask) {
  int score = 0;
  for (int row = 0; row < 8; row++) {
    int seats = (seatingMask >> (row * 4)) & 15;
    int occupied = 0;
    for (int b = 0; b < 4; b++) {
      if (((seats >> b) & 1) == 1) occupied++;
    }
    bool leftAisle = (seats & 1) != 0;
    bool rightAisle = (seats & 8) != 0;
    if (leftAisle && rightAisle) {
      score += occupied * 2 + (occupied == 4 ? 3 : 1);
    } else if (leftAisle || rightAisle) {
      score += occupied - (occupied == 1 ? 1 : 0);
    } else if (occupied >= 3) {
      score += 1;
    } else {
      score -= occupied;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(evaluateTheaterAislePressure(0) == 0);
  assert(evaluateTheaterAislePressure(15) == 11);
  assert(evaluateTheaterAislePressure(159) == 16);
  print('All tests passed!');
}