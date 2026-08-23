@pragma('vm:entry-point')
List<String> rebuildAsciiBannerRows(List<String> baseRows, List<String> actions, int maxWidth) {
  var rows = List<String>.from(baseRows);
  for (var action in actions) {
    if (action == '^') {
      if (rows.isNotEmpty) rows.removeLast();
    } else if (action.length <= maxWidth) {
      rows.add(action);
    }
  }
  return rows;
}

@pragma('vm:entry-point')
void main() {
  assert(rebuildAsciiBannerRows(['==-'], ['***'], 3).toString() == '[==-, ***]');
  assert(rebuildAsciiBannerRows([], ['^^^^'], 2).length == 0);
  assert(rebuildAsciiBannerRows(['<>'], ['^', '--'], 2).toString() == '[--]');
  print('All tests passed!');
}