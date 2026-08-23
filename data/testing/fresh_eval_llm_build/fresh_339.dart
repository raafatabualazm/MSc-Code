@pragma('vm:entry-point')
int thermostatCyclePenalty(int schedule) {
  int score = 0;
  int prev = schedule & 1;
  for (int i = 1; i < 24; i++) {
    int bit = (schedule >> i) & 1;
    if (bit != prev) {
      score += 2;
      if (((schedule >> ((i + 1) % 24)) & 1) == prev) {
        score += 3;
      } else {
        score += 1;
      }
    } else if (bit == 1 && (i % 6 == 5)) {
      score--;
    }
    prev = bit;
  }
  if ((((schedule >> 23) & 1) ^ (schedule & 1)) == 1) {
    score += 4;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(thermostatCyclePenalty(0) == 0);
  assert(thermostatCyclePenalty(5) == 17);
  assert(thermostatCyclePenalty(63) == 6);
  print('All tests passed!');
}