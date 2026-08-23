@pragma('vm:never-inline')
@pragma('vm:entry-point')
double candidate(String barcode) {
  double oddSum = 0.0;
  double evenSum = 0.0;
  bool isOdd = true;
  for (int i = 0; i < barcode.length; i++) {
    final int digit = barcode.codeUnitAt(i) - 48;
    if (isOdd) {
      oddSum += digit;
    } else {
      evenSum += digit;
    }
    isOdd = !isOdd;
  }
  return oddSum - evenSum;
}

void main() {
  final implementation = candidate;

  expect(implementation('1234'), -2.0);
  expect(implementation('8642'), 4.0);
  expect(implementation('1111'), 0.0);
  expect(implementation(''), 0.0);
  expect(implementation('5'), 5.0);
  expect(implementation('09'), -9.0);
  expect(implementation('246'), 4.0);
  expect(implementation('00'), 0.0);
  expect(implementation('9000'), 9.0);
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
