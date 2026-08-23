@pragma('vm:never-inline')
@pragma('vm:entry-point')
bool candidate(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {

  if (ax2 <= bx1 || bx2 <= ax1) return false;
  if (ay2 <= by1 || by2 <= ay1) return false;
  return true;
}

void main() {
  final implementation = candidate;

  expect(implementation(0, 0, 2, 2, 1, 1, 3, 3), true);
  expect(implementation(0, 0, 1, 1, 2, 2, 3, 3), false);
  expect(implementation(0, 0, 2, 2, 2, 0, 4, 2), false);
  expect(implementation(0, 0, 3, 3, 1, 1, 2, 2), true);
  expect(implementation(0, 0, 1, 4, 0, 2, 1, 6), true);
  expect(implementation(5, 5, 10, 10, 0, 0, 5, 5), false);
  expect(implementation(0, 0, 4, 4, 3, 3, 7, 7), true);
  expect(implementation(0, 0, 1, 1, 1, 0, 2, 1), false);
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
