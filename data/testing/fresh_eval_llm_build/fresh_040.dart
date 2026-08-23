@pragma('vm:entry-point')
bool doesTopPrecinctReachMajorityFirst(List<int> tallies) {
  if (tallies.isEmpty) return false;
  int total = 0;
  for (var t in tallies) total += t;
  if (total == 0) return false;
  int threshold = total ~/ 2 + 1;
  int sum = 0;
  List<int> pref = [];
  for (var t in tallies) {
    sum += t;
    pref.add(sum);
  }
  int l = 0, r = pref.length;
  while (l < r) {
    int m = (l + r) ~/ 2;
    if (pref[m] < threshold) l = m + 1;
    else r = m;
  }
  int majIdx = l;
  int maxIdx = 0, maxVal = tallies[0];
  for (int i = 1; i < tallies.length; i++) {
    if (tallies[i] > maxVal) {
      maxVal = tallies[i];
      maxIdx = i;
    }
  }
  return majIdx == maxIdx;
}

@pragma('vm:entry-point')
void main() {
  assert(doesTopPrecinctReachMajorityFirst([]) == false);
  assert(doesTopPrecinctReachMajorityFirst([5]) == true);
  assert(doesTopPrecinctReachMajorityFirst([3, 2, 1]) == false);
  print('All tests passed!');
}