@pragma('vm:entry-point')
List<int> compactRoundedNickelCents(List<int> centsValues) {
  List<int> result = [];
  for (final cents in centsValues) {
    int rounded = cents >= 0
        ? ((cents + 2) ~/ 5) * 5
        : ((cents - 2) ~/ 5) * 5;
    if (result.isEmpty || result.last != rounded) {
      result.add(rounded);
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(compactRoundedNickelCents([]).length == 0);
  assert(compactRoundedNickelCents([3, 4, 5]).toString() == '[5]');
  assert(compactRoundedNickelCents([-2, -3, -8]).toString() == '[0, -5, -10]');
  print('All tests passed!');
}