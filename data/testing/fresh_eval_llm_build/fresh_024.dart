@pragma('vm:entry-point')
bool playlistRunChecksumValid(String tape) {
  int hash = tape.indexOf('#');
  if (hash <= 0 || hash != tape.lastIndexOf('#') || hash != tape.length - 2) {
    return false;
  }
  int total = 0;
  String previous = '';
  for (int i = 0; i < hash; i += 2) {
    if (i + 1 >= hash) return false;
    String song = tape[i];
    int count = tape.codeUnitAt(i + 1) - 48;
    if (!RegExp(r'[A-Z]').hasMatch(song) || count < 1 || count > 9) {
      return false;
    }
    if (song == previous) {
      if (count.isOdd) return false;
      total += count ~/ 2;
    } else {
      total += 'AEIOU'.contains(song) ? count * 2 : count;
    }
    previous = song;
  }
  return total % 10 == tape.codeUnitAt(hash + 1) - 48;
}

@pragma('vm:entry-point')
void main() {
  assert(playlistRunChecksumValid('A1#2') == true);
  assert(playlistRunChecksumValid('B3B2#4') == true);
  assert(playlistRunChecksumValid('B3B3#6') == false);
  print('All tests passed!');
}