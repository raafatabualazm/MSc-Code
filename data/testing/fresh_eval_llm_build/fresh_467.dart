@pragma('vm:entry-point')
bool validateDiceRoundFlow(String log) {
  if (log.isEmpty) return true;
  var rounds = log.split('|');
  for (var round in rounds) {
    if (round.isEmpty) return false;
    var parts = round.split(',');
    var sum = 0;
    var sawRoll = false;
    var ended = false;
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i];
      if (p.length == 2 && p.codeUnitAt(0) == 68) {
        var v = p.codeUnitAt(1) - 48;
        if (v < 1 || v > 6 || ended) return false;
        sum += v;
        sawRoll = true;
        if (sum > 12 && (i + 1 >= parts.length || parts[i + 1] != 'X')) return false;
        continue;
      }
      if ((p == 'H' || p == 'X') && !ended) {
        if (!sawRoll || i != parts.length - 1) return false;
        ended = true;
        if (p == 'H') {
          if (sum < 7 || sum > 12) return false;
        } else if (sum <= 12) {
          return false;
        }
        continue;
      }
      return false;
    }
    if (!ended) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(validateDiceRoundFlow('D3,D4,H') == true);
  assert(validateDiceRoundFlow('D6,D6,X') == false);
  assert(validateDiceRoundFlow('D6,D6,D1,X|D2,D5,H') == true);
  print('All tests passed!');
}