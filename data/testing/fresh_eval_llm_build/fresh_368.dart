@pragma('vm:entry-point')
int scoreBracketTokenChaos(String bracketLog) {
  int score = 0;
  for (String part in bracketLog.split('|')) {
    if (part.isEmpty) {
      score -= 1;
      continue;
    }
    List<String> halves = part.split('>');
    if (halves.length != 2 || !halves[0].contains('/')) {
      score += 5;
    } else {
      List<String> seeds = halves[0].split('/');
      int? left = int.tryParse(seeds[0]);
      int? right = int.tryParse(seeds[1]);
      String winner = halves[1];
      if (left == null || right == null || (winner != 'L' && winner != 'R')) {
        score += 4;
      } else if ((winner == 'L' && left > right) || (winner == 'R' && right > left)) {
        score += 3;
      } else if (left == right) {
        score += 1;
      } else {
        score -= 2;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(scoreBracketTokenChaos('16/1>L') == 3);
  assert(scoreBracketTokenChaos('1/16>L|8/8>R') == -1);
  assert(scoreBracketTokenChaos('||') == -3);
  print('All tests passed!');
}