@pragma('vm:entry-point')
List<int> computeBonusDays(List<int> diceRolls) {
  List<int> bonusDays = [];
  int currentDay = 0;
  for (int i = 0; i < diceRolls.length; i++) {
    int roll = diceRolls[i];
    if (roll > 0) {
      currentDay += roll;
      if (currentDay % 13 != 0) {
        if (i == 0 && currentDay == 7) {
          bonusDays.add(currentDay);
        } else if (currentDay % 7 == 0 && (roll == 2 || roll == 3 || roll == 5)) {
          bonusDays.add(currentDay);
        }
      }
    }
  }
  return bonusDays;
}

@pragma('vm:entry-point')
void main() {
  assert(computeBonusDays([]).isEmpty);
  assert(computeBonusDays([7]).length == 1);
  assert(computeBonusDays([7, 5, 2])[1] == 14);
  print('All tests passed!');
}