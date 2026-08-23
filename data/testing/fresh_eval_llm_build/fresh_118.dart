@pragma('vm:entry-point')
bool hasMazeRowVisitedPassage(String row) {
  if (row.isEmpty) return false;
  final tokens = row.split('|');
  bool foundVisitedPassage = false;
  const validWalls = {'N', 'S', 'E', 'W', 'X'};
  for (final token in tokens) {
    if (token.length != 3) return false;
    final passage = token[0];
    final wall = token[1];
    final countChar = token[2];
    if (passage != '.' && passage != '#') return false;
    if (!validWalls.contains(wall)) return false;
    final count = int.tryParse(countChar);
    if (count == null) return false;
    if (passage == '.' && count > 0) foundVisitedPassage = true;
  }
  return foundVisitedPassage;
}

@pragma('vm:entry-point')
void main() {
  assert(hasMazeRowVisitedPassage('.N1|#S0') == true);
  assert(hasMazeRowVisitedPassage('#N1|#S0') == false);
  assert(hasMazeRowVisitedPassage('') == false);
  print('All tests passed!');
}