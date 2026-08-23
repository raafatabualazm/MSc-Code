@pragma('vm:entry-point')
String? pickBalancedWifiBin(List<int> signals, List<String> bins, bool preferStronger) {
  if (signals.isEmpty || signals.length != bins.length) return null;
  final order = List<int>.generate(signals.length, (i) => i);
  order.sort((a, b) {
    final da = (signals[a] + 70).abs();
    final db = (signals[b] + 70).abs();
    if (da != db) return da - db;
    return preferStronger ? signals[b] - signals[a] : signals[a] - signals[b];
  });
  return bins[order.first];
}

@pragma('vm:entry-point')
void main() {
  assert(pickBalancedWifiBin([-70], ['stable'], true) == 'stable');
  assert(pickBalancedWifiBin([-69, -71], ['near', 'far'], false) == 'far');
  assert(pickBalancedWifiBin([], [], true) == null);
  print('All tests passed!');
}