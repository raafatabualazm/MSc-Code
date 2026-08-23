@pragma('vm:entry-point')
int countRelayEdgePulses(List<String> panel) {
  int score = 0;
  for (final row in panel) {
    if (row.isNotEmpty && row[0] == '-' && row[row.length - 1] == '.') {
      score += row.length;
    } else if (row.contains('..-')) {
      score -= 1;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(countRelayEdgePulses([]) == 0);
  assert(countRelayEdgePulses(['-.', '--.']) == 5);
  assert(countRelayEdgePulses(['..-', '-..']) == 2);
  print('All tests passed!');
}