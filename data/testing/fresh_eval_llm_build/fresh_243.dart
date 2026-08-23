@pragma('vm:entry-point')
int countUnsyncedTracks(List<int> trackLengths) {
  if (trackLengths.isEmpty) return 0;
  int total = 0;
  for (int t in trackLengths) {
    total += t;
  }
  if (total == 0) return 0;
  int count = 0;
  for (int t in trackLengths) {
    int a = t < 0 ? -t : t;
    int b = total;
    while (b != 0) {
      int temp = b;
      b = a % b;
      a = temp;
    }
    if (a == 1) count++;
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countUnsyncedTracks([]) == 0);
  assert(countUnsyncedTracks([3, 5, 7]) == 1);
  assert(countUnsyncedTracks([1, 2, 3]) == 1);
  print('All tests passed!');
}