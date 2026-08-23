@pragma('vm:entry-point')
bool verifyBracketChecksum(String bracket, int expectedChecksum) {
  if (bracket.length % 2 != 0) return false;
  int sum = 0;
  for (int i = 0; i < bracket.length; i += 2) {
    int seed = bracket.codeUnitAt(i) - 'A'.codeUnitAt(0) + 1;
    int points = int.parse(bracket[i+1]);
    sum += (seed + points) * ((i ~/ 2) + 1);
  }
  return sum == expectedChecksum;
}

@pragma('vm:entry-point')
void main() {
  assert(verifyBracketChecksum('A5', 6) == true);
  assert(verifyBracketChecksum('', 0) == true);
  assert(verifyBracketChecksum('A5B3', 16) == true);
  print('All tests passed!');
}