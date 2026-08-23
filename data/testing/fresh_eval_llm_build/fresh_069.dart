@pragma('vm:entry-point')
List<int> possibleFinalRarities(List<int> rarities) {
  if (rarities.isEmpty) return [];
  if (rarities.length == 1) return [rarities[0]];
  Set<int> results = {};
  for (int i = 1; i < rarities.length; i++) {
    List<int> left = rarities.sublist(0, i);
    List<int> right = rarities.sublist(i);
    List<int> leftResults = possibleFinalRarities(left);
    List<int> rightResults = possibleFinalRarities(right);
    for (int l in leftResults) {
      for (int r in rightResults) {
        if ((l + r) % 2 == 0) {
          results.add((l + r) ~/ 2);
        } else {
          results.add(l > r ? l : r);
        }
      }
    }
  }
  var sorted = results.toList();
  sorted.sort();
  return sorted;
}

@pragma('vm:entry-point')
void main() {
  assert(possibleFinalRarities([1,2]).toString() == '[2]');
  assert(possibleFinalRarities([1,2,3]).length == 2);
  assert(possibleFinalRarities([]).isEmpty);
  print('All tests passed!');
}