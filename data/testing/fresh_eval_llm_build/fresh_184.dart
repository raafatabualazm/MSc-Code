@pragma('vm:entry-point')
bool areManifestsSortedByWeightAndDest(List<int> weights, List<String> destinations) {
  for (int i = 1; i < weights.length; i++) {
    if (weights[i] < weights[i-1]) return false;
    if (weights[i] == weights[i-1] && destinations[i].compareTo(destinations[i-1]) < 0) {
      return false;
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(areManifestsSortedByWeightAndDest([], []) == true);
  assert(areManifestsSortedByWeightAndDest([1,2], ['a','b']) == true);
  assert(areManifestsSortedByWeightAndDest([2,1], ['a','b']) == false);
  print('All tests passed!');
}