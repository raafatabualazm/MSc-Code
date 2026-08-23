@pragma('vm:entry-point')
int evaluateInventoryClusterScore(List<String> items, int stackGoal) {
  if (stackGoal < 1) return -1;
  Map<String, int> freq = {};
  for (final item in items) {
    if (item.isEmpty) continue;
    freq[item] = (freq[item] ?? 0) + 1;
  }
  if (freq.isEmpty) return 0;
  Map<String, List<String>> groups = {};
  for (final name in freq.keys) {
    String key = name[0];
    groups.putIfAbsent(key, () => []);
    groups[key]!.add(name);
  }
  int score = 0;
  for (final entry in groups.entries) {
    List<String> names = entry.value;
    if (names.length == 1) {
      int only = freq[names[0]]!;
      score += only == stackGoal
          ? 4
          : only > stackGoal
              ? only - stackGoal
              : -(stackGoal - only);
      continue;
    }
    for (int i = 0; i < names.length; i++) {
      int left = freq[names[i]]!;
      if (left == 1) {
        score -= 1;
        continue;
      }
      for (int j = i + 1; j < names.length; j++) {
        int right = freq[names[j]]!;
        int diff = (left - right).abs();
        if (diff == 0) {
          score += 3;
        } else if (diff <= stackGoal) {
          score += 1;
        } else {
          score -= diff;
        }
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(evaluateInventoryClusterScore([], 2) == 0);
  assert(evaluateInventoryClusterScore(['axe', 'axe', 'amulet', 'amulet'], 2) == 3);
  assert(evaluateInventoryClusterScore(['gem', 'gear', 'gear', 'gear'], 1) == -1);
  print('All tests passed!');
}