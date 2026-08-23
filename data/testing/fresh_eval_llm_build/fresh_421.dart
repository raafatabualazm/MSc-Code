@pragma('vm:entry-point')
int scoreMezzanineSeatTurnover(List<int> openSeatsByDay) {
  int score = 0;
  int lastShuffleDay = 0;
  for (int day = 0; day < openSeatsByDay.length; day++) {
    int seats = openSeatsByDay[day];
    int age = day - lastShuffleDay;
    if (seats < 0) {
      score -= seats.abs() * (age + 1);
      lastShuffleDay = day;
    } else if (seats == 0) {
      score += age;
    } else {
      if (seats % 2 == 0) {
        score += seats ~/ (age + 1);
      } else {
        score += seats - age;
      }
    }
  }
  for (int day = 1; day < openSeatsByDay.length; day++) {
    int diff = openSeatsByDay[day] - openSeatsByDay[day - 1];
    if (diff.abs() >= 3) {
      score += day;
    } else if (diff == 0) {
      score -= 1;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(scoreMezzanineSeatTurnover([]) == 0);
  assert(scoreMezzanineSeatTurnover([2, 2]) == 2);
  assert(scoreMezzanineSeatTurnover([0, -3, 0]) == -2);
  print('All tests passed!');
}