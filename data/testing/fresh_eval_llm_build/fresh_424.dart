@pragma('vm:entry-point')
int majorityVoteGap(List<String> votes) {
  if (votes.isEmpty) return 0;
  final tally = <String, int>{};
  for (final v in votes) {
    tally[v] = (tally[v] ?? 0) + 1;
  }
  int maxVotes = -1;
  bool tie = false;
  for (final entry in tally.entries) {
    if (entry.value > maxVotes) {
      maxVotes = entry.value;
      tie = false;
    } else if (entry.value == maxVotes) {
      tie = true;
    }
  }
  if (tie) return -1;
  final majority = votes.length ~/ 2 + 1;
  if (maxVotes >= majority) return 0;
  return majority - maxVotes;
}

@pragma('vm:entry-point')
void main() {
  assert(majorityVoteGap([]) == 0);
  assert(majorityVoteGap(['A']) == 0);
  assert(majorityVoteGap(['A', 'B']) == -1);
  print('All tests passed!');
}