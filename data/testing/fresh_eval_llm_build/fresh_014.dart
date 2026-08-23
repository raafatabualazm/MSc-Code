@pragma('vm:entry-point')
List<int> summarizeDiceSwingRounds(List<int> rolls, int pivot) {
  if (rolls.isEmpty) return [];
  List<int> result = [];
  List<int> round = [];
  for (int i = 0; i <= rolls.length; i++) {
    if (i < rolls.length && rolls[i] != 0) {
      round.add(rolls[i]);
      continue;
    }
    if (round.isEmpty) continue;
    List<int> counts = List.filled(7, 0);
    int sum = 0;
    bool invalid = false;
    for (int value in round) {
      if (value < 1 || value > 6) {
        invalid = true;
        break;
      }
      counts[value]++;
      sum += value;
    }
    if (invalid) {
      round = [];
      continue;
    }
    int distinct = 0;
    int highFace = 0;
    bool heavy = false;
    for (int face = 1; face <= 6; face++) {
      if (counts[face] > 0) {
        distinct++;
        highFace = face;
      }
      if (counts[face] >= pivot) heavy = true;
    }
    int streak = 1;
    bool run = false;
    for (int j = 1; j < round.length; j++) {
      if (round[j] == round[j - 1] + 1) {
        streak++;
        if (streak >= 3) run = true;
      } else {
        streak = 1;
      }
    }
    int score = sum;
    if (run) score -= pivot;
    score += heavy ? distinct : -distinct;
    if (distinct == 2 && round.length > pivot) score += highFace;
    if (result.isEmpty || result.last != score) result.add(score);
    round = [];
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(summarizeDiceSwingRounds([], 2).isEmpty);
  assert(summarizeDiceSwingRounds([1,2,0,2,1], 2).length == 1);
  assert(summarizeDiceSwingRounds([2,2,5,5,0,1,1,1], 2).toString() == '[21, 4]');
  print('All tests passed!');
}