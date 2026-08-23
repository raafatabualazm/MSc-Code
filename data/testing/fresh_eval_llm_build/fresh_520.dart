@pragma('vm:entry-point')
bool validateBracketProgression(String bracket) {
  if (bracket.isEmpty) return false;
  List<String> rounds = bracket.split(';');
  Set<String> previousWinners = {};
  for (int i = 0; i < rounds.length; i++) {
    String round = rounds[i].trim();
    if (round.isEmpty) return false;
    List<String> matches = round.split(',');
    Set<String> players = {};
    Set<String> winners = {};
    for (String match in matches) {
      String token = match.trim();
      List<String> sides = token.split('>');
      if (sides.length != 2) return false;
      String winner = sides[0].trim();
      String loser = sides[1].trim();
      if (winner.isEmpty || loser.isEmpty || winner == loser) return false;
      if (players.contains(winner) || players.contains(loser)) return false;
      players.add(winner);
      players.add(loser);
      winners.add(winner);
    }
    if (previousWinners.isNotEmpty) {
      if (players.length != previousWinners.length) return false;
      for (String p in players) {
        if (!previousWinners.contains(p)) return false;
      }
    }
    if (i < rounds.length - 1 && winners.length < 2) return false;
    previousWinners = winners;
  }
  return previousWinners.length == 1;
}

@pragma('vm:entry-point')
void main() {
  assert(validateBracketProgression('A>B') == true);
  assert(validateBracketProgression('A>B,C>D;B>C') == false);
  assert(validateBracketProgression('A>B,C>D;A>C') == true);
  print('All tests passed!');
}