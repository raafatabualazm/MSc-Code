@pragma('vm:entry-point')
int countEvenTiedVoteTallies(List<String> votes) {
  Map<String, int> tally = {};
  for (String candidate in votes) {
    if (candidate.isEmpty) continue;
    tally[candidate] = (tally[candidate] ?? 0) + 1;
  }
  Map<int, int> countFreq = {};
  for (int cnt in tally.values) {
    countFreq[cnt] = (countFreq[cnt] ?? 0) + 1;
  }
  int evenTies = 0;
  for (int cnt in countFreq.keys) {
    int numCandidates = countFreq[cnt]!;
    if (numCandidates >= 2 && cnt % 2 == 0) {
      evenTies++;
    }
  }
  return evenTies;
}

@pragma('vm:entry-point')
void main() {
  assert(countEvenTiedVoteTallies([]) == 0);
  assert(countEvenTiedVoteTallies(['Alice','Bob','Alice','Bob']) == 1);
  assert(countEvenTiedVoteTallies(['Alice','Bob','Charlie']) == 0);
  print('All tests passed!');
}