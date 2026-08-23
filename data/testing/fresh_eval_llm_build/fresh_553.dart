@pragma('vm:entry-point')
String summarizeSeatAisles(String chart) {
  if (chart.isEmpty) return 'empty';
  List<String> out = [];
  for (String row in chart.split('|')) {
    if (row.isEmpty) continue;
    int colon = row.indexOf(':');
    if (colon <= 0 || colon == row.length - 1) return 'invalid';
    String name = row.substring(0, colon);
    int streak = 0, maxStreak = 0, blocked = 0, lastSeat = 0;
    bool disorder = false;
    for (String token in row.substring(colon + 1).split(',')) {
      if (token.isEmpty) continue;
      String kind = token[0];
      if (kind == '_') {
        streak = 0;
        continue;
      }
      if (token.length < 2) return 'invalid';
      int? seat = int.tryParse(token.substring(1));
      if (seat == null || seat <= 0) return 'invalid';
      if (seat <= lastSeat && (kind == 'R' || kind == 'V')) disorder = true;
      lastSeat = seat;
      if (kind == 'X') {
        blocked++;
        streak = 0;
      } else if (kind == 'R' || kind == 'V') {
        streak++;
        if (streak > maxStreak) maxStreak = streak;
      } else {
        return 'invalid';
      }
    }
    out.add('$name=$maxStreak/$blocked${disorder ? '!' : ''}');
  }
  return out.isEmpty ? 'empty' : out.join(';');
}

@pragma('vm:entry-point')
void main() {
  assert(summarizeSeatAisles('A:R1,V2,_,R3') == 'A=2/0');
  assert(summarizeSeatAisles('B:R3,V2,R4') == 'B=3/0!');
  assert(summarizeSeatAisles('||') == 'empty');
  print('All tests passed!');
}