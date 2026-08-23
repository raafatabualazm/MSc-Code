@pragma('vm:entry-point')
bool isAnyCandidateDominated(List<Map<String, int>> precincts, int minDominatorVotes) {
  Set<String> candidatesSet = {};
  Map<String, int> totalVotes = {};
  for (var precinct in precincts) {
    for (var entry in precinct.entries) {
      String candidate = entry.key;
      int votes = entry.value;
      candidatesSet.add(candidate);
      totalVotes[candidate] = (totalVotes[candidate] ?? 0) + votes;
    }
  }
  List<String> candidates = candidatesSet.toList();
  for (String x in candidates) {
    for (String y in candidates) {
      if (x == y) continue;
      if ((totalVotes[y] ?? 0) < minDominatorVotes) continue;
      bool dominated = true;
      bool strictlyGreater = false;
      for (var precinct in precincts) {
        int vX = precinct[x] ?? 0;
        int vY = precinct[y] ?? 0;
        if (vX > vY) {
          dominated = false;
          break;
        }
        if (vY > vX) {
          strictlyGreater = true;
        }
      }
      if (dominated && strictlyGreater) {
        return true;
      }
    }
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(isAnyCandidateDominated([], 0) == false);
  assert(isAnyCandidateDominated([{'A': 5, 'B': 10}], 10) == true);
  assert(isAnyCandidateDominated([{'A': 5, 'B': 10}], 11) == false);
  print('All tests passed!');
}