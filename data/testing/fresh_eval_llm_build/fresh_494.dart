@pragma('vm:entry-point')
int countMatchingTrafficObservations(String observations, List<int> durations) {
  if (durations.length != 3 || durations.any((d) => d <= 0)) return -1;
  int phase = 0, remaining = durations[0], matches = 0;
  for (int i = 0; i < observations.length; i++) {
    String obs = observations[i];
    if (obs != 'R' && obs != 'G' && obs != 'Y') return -1;
    String expected = phase == 0 ? 'R' : phase == 1 ? 'G' : 'Y';
    if (obs == expected) matches++;
    if (--remaining == 0) {
      phase = (phase + 1) % 3;
      remaining = durations[phase];
    }
  }
  return matches;
}

@pragma('vm:entry-point')
void main() {
  assert(countMatchingTrafficObservations("", [1,2,3]) == 0);
  assert(countMatchingTrafficObservations("R", [1,1,1]) == 1);
  assert(countMatchingTrafficObservations("RRRGGY", [3,2,1]) == 6);
  print('All tests passed!');
}