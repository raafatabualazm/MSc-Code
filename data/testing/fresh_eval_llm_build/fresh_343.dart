@pragma('vm:entry-point')
String summarizeInventoryTiers(List<String> items) {
  Map<String, int> freq = {};
  for (var item in items) {
    freq[item] = (freq[item] ?? 0) + 1;
  }
  List<String> legendary = [];
  List<String> rare = [];
  List<String> common = [];
  for (var entry in freq.entries) {
    if (entry.value >= 4) {
      legendary.add(entry.key);
    } else if (entry.value >= 2) {
      rare.add(entry.key);
    } else {
      common.add(entry.key);
    }
  }
  legendary.sort();
  rare.sort();
  common.sort();
  return 'Legendary: [${legendary.join(', ')}], Rare: [${rare.join(', ')}], Common: [${common.join(', ')}]';
}

@pragma('vm:entry-point')
void main() {
  assert(summarizeInventoryTiers([]) == "Legendary: [], Rare: [], Common: []");
  assert(summarizeInventoryTiers(['sword']) == "Legendary: [], Rare: [], Common: [sword]");
  assert(summarizeInventoryTiers(['a','a','a','a']) == "Legendary: [a], Rare: [], Common: []");
  print('All tests passed!');
}