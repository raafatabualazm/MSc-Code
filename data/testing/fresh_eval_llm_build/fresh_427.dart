@pragma('vm:entry-point')
int longestMazeTorchChain(List<String> grid) {
  int best = 0, streak = 0, prev = -100;
  for (final row in grid) {
    int torch = row.indexOf('T');
    if (torch >= 0 && (prev < 0 || (torch - prev).abs() <= 1)) {
      streak++;
    } else {
      streak = torch >= 0 ? 1 : 0;
    }
    if (streak > best) best = streak;
    prev = torch;
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(longestMazeTorchChain([]) == 0);
  assert(longestMazeTorchChain(['T..', '.T.', '..T']) == 3);
  assert(longestMazeTorchChain(['T..', '..T', 'T..']) == 1);
  print('All tests passed!');
}