@pragma('vm:entry-point')
String selectCriticalLogBranch(List<String> logs) {
  if (logs.isEmpty) return 'empty';
  final children = <String, List<String>>{};
  final state = <String, String>{};
  final roots = <String>[];
  for (final line in logs) {
    final parts = line.split(' ');
    if (parts.length != 3) continue;
    final name = parts[0], parent = parts[1], level = parts[2];
    state[name] = level;
    if (parent == '-') {
      roots.add(name);
    } else {
      children.putIfAbsent(parent, () => <String>[]).add(name);
    }
  }
  String best = 'empty';
  int bestScore = -1;
  void dfs(String node, List<String> path, int score) {
    final level = state[node] ?? 'OK';
    if (level == 'DROP') return;
    final next = score + (level == 'ERR' ? 3 : (level == 'WARN' ? 1 : 0));
    path.add(node);
    final kids = children[node] ?? <String>[];
    if (kids.isEmpty) {
      final joined = path.join('->');
      if (next > bestScore || (next == bestScore && (best == 'empty' || path.length < best.split('->').length || (path.length == best.split('->').length && joined.compareTo(best) < 0)))) {
        bestScore = next;
        best = joined;
      }
    } else {
      for (final child in kids) {
        if ((state[child] ?? '') == 'DROP' && kids.length > 1) continue;
        dfs(child, List<String>.from(path), next);
      }
    }
  }
  for (final root in roots) {
    dfs(root, <String>[], 0);
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(selectCriticalLogBranch([]) == 'empty');
  assert(selectCriticalLogBranch(['api - ERR']) == 'api');
  assert(selectCriticalLogBranch(['a - OK','b a WARN','c b ERR']) == 'a->b->c');
  print('All tests passed!');
}