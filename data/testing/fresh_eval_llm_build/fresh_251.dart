@pragma('vm:entry-point')
int computeBracketVolatility(List<int> winners, int eliteSeed) {
  Set<int> uniqueSeeds = {};
  int score = 0;
  for (int seed in winners) {
    bool firstTime = uniqueSeeds.add(seed);
    if (seed > eliteSeed) {
      if (firstTime) {
        score += seed - eliteSeed;
      } else {
        score += 1;
      }
    } else {
      if (firstTime && eliteSeed - seed <= 2) {
        score += 2;
      } else if (!firstTime) {
        score -= 1;
      }
    }
  }
  for (int seed in uniqueSeeds) {
    if (seed.isEven) {
      score += 1;
    } else {
      score -= 1;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(computeBracketVolatility([], 5) == 0);
  assert(computeBracketVolatility([6, 4, 4, 9], 4) == 9);
  assert(computeBracketVolatility([2, 2], 5) == 0);
  print('All tests passed!');
}