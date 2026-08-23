@pragma('vm:entry-point')
int countBalancedPlaylistCuts(List<int> energyLevels, int tolerance) {
  int solve(List<int> part) {
    if (part.length < 2) return 0;
    int mid = part.length ~/ 2;
    int left = part.sublist(0, mid).fold(0, (a, b) => a + b);
    int right = part.sublist(mid).fold(0, (a, b) => a + b);
    return (left - right).abs() <= tolerance
        ? 1 + solve(part.sublist(0, mid)) + solve(part.sublist(mid))
        : solve(part.sublist(0, mid)) + solve(part.sublist(mid));
  }

  return solve(energyLevels);
}

@pragma('vm:entry-point')
void main() {
  assert(countBalancedPlaylistCuts([], 0) == 0);
  assert(countBalancedPlaylistCuts([1, 2, 3, 4], 3) == 2);
  assert(countBalancedPlaylistCuts([2, 2, 2, 2, 2, 2, 2, 2], 0) == 7);
  print('All tests passed!');
}