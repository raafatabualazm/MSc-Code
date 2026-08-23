@pragma('vm:entry-point')
List<int> compressTrafficPhases(String phases) {
  if (phases.isEmpty) return [];
  var out = <int>[];
  var count = 1;
  for (var i = 1; i <= phases.length; i++) {
    if (i < phases.length && phases[i] == phases[i - 1]) {
      count++;
    } else {
      var c = phases[i - 1];
      out.add((c == 'R' ? 100 : c == 'Y' ? 200 : 300) + count);
      count = 1;
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(compressTrafficPhases('').toString() == '[]');
  assert(compressTrafficPhases('RRG').toString() == '[102, 301]');
  assert(compressTrafficPhases('GGYYR').toString() == '[302, 202, 101]');
  print('All tests passed!');
}