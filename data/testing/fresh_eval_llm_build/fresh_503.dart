@pragma('vm:entry-point')
List<int> filterThenScoreDiceRounds(List<List<int>> rounds, int banned) {
  List<int> results = [];
  for (var dice in rounds) {
    int sum = 0;
    int maxDie = -1;
    for (var die in dice) {
      if (die != banned) {
        sum += die;
        if (die > maxDie) maxDie = die;
      }
    }
    int n = dice.length;
    int score;
    if (n == 2 && dice[0] == dice[1]) {
      score = sum * 2;
    } else if (n == 3 && dice[0] == dice[1] && dice[1] == dice[2]) {
      score = sum * 3;
    } else if (n > 3) {
      score = maxDie == -1 ? 0 : maxDie;
    } else {
      score = sum;
    }
    results.add(score);
  }
  return results;
}

@pragma('vm:entry-point')
void main() {
  assert(filterThenScoreDiceRounds([], 1).toString() == '[]');
  assert(filterThenScoreDiceRounds([[5], [3,3], [1,2,3,4]], 1).toString() == '[5, 12, 4]');
  assert(filterThenScoreDiceRounds([[2,2],[2,2,2],[2,2,2,2]], 1).toString() == '[8, 18, 2]');
  print('All tests passed!');
}