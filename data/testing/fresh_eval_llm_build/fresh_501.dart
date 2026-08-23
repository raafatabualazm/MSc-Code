@pragma('vm:entry-point')
String planBatteryCycleWindows(List<int> cycles, int reserve) {
  if (cycles.isEmpty) return 'idle';
  String best = '';
  int bestCount = -1;

  void search(int start, List<String> parts) {
    if (start == cycles.length) {
      String joined = parts.join('|');
      if (parts.length > bestCount ||
          (parts.length == bestCount &&
              (best.isEmpty || joined.compareTo(best) < 0))) {
        bestCount = parts.length;
        best = joined;
      }
      return;
    }
    for (int end = start; end < cycles.length; end++) {
      int sum = 0;
      bool unstable = false;
      for (int i = start; i <= end; i++) {
        sum += cycles[i];
        if (i > start && (cycles[i] - cycles[i - 1]).abs() > reserve) {
          unstable = true;
          break;
        }
      }
      if (unstable) continue;
      if (sum < reserve) continue;
      if (sum > reserve * 2) break;
      parts.add('$start-$end');
      search(end + 1, parts);
      parts.removeLast();
    }
  }

  search(0, []);
  return best.isEmpty ? 'drain' : best;
}

@pragma('vm:entry-point')
void main() {
  assert(planBatteryCycleWindows([], 3) == 'idle');
  assert(planBatteryCycleWindows([2, 1, 2], 2) == '0-0|1-2');
  assert(planBatteryCycleWindows([1, 5], 3) == 'drain');
  print('All tests passed!');
}