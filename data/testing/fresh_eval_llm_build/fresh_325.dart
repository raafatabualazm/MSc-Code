@pragma('vm:entry-point')
List<String> matchingDockFootprints(List<String> crates) {
  var matches = <String>[];
  for (var crate in crates) {
    var p = crate.split(',');
    var area = (int.parse(p[3]) - int.parse(p[1])) * (int.parse(p[4]) - int.parse(p[2]));
    var distance = int.parse(p[1]).abs() + int.parse(p[2]).abs();
    if (area == distance) matches.add(p[0]);
  }
  return matches;
}

@pragma('vm:entry-point')
void main() {
  assert(matchingDockFootprints([]).toString() == '[]');
  assert(matchingDockFootprints(['boxA,1,2,4,3']).toString() == '[boxA]');
  assert(matchingDockFootprints(['boxB,-1,1,1,2']).length == 1);
  print('All tests passed!');
}