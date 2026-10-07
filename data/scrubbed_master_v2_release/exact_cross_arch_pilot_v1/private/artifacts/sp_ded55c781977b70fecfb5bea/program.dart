@pragma('vm:never-inline')
@pragma('vm:entry-point')
int candidate(List<int> centAmounts) {
  if (centAmounts.isEmpty) return -1;
  int centDist(int v) {
    int r = v % 100;
    return r < 50 ? r : 100 - r;
  }
  List<int> sorted = List<int>.from(centAmounts);
  sorted.sort((a, b) => centDist(a).compareTo(centDist(b)));
  return sorted.first;
}

void main() {
  final implementation = candidate;

  expect(implementation([149, 251, 300, 175]), 300);
  expect(implementation([110, 190, 155]), 110);
  expect(implementation([]), -1);
  expect(implementation([50]), 50);
  expect(implementation([100, 199, 205]), 100);
  expect(implementation([133, 167, 150]), 133);
  expect(implementation([301, 449, 400]), 400);
  expect(implementation([275, 225, 250]), 275);
}

void expect(dynamic a, dynamic b) {
  if (a == b) return;

  if (a is List && b is List) {
    expectList(a, b);
  } else if (a is Map && b is Map) {
    expectMap(a, b);
  } else {
    throw '$a != $b';
  }
}

void expectList(List a, List b) {
  if (a.length != b.length) throw 'list lengths are not equal';

  for (var i = 0; i < a.length; i++) {
    expect(a[i], b[i]);
  }
}

void expectMap(Map a, Map b) {
  if (a.length != b.length) throw 'map lengths are not equal';

  for (var key in a.keys) {
    expect(a[key], b[key]);
  }
}
