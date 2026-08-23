@pragma('vm:entry-point')
List<String> collectNarrowDigitBarcodes(List<String> barcodes, int distinctLimit) {
  return barcodes
      .where((b) => b.split('').toSet().length <= distinctLimit)
      .map((b) => b.isEmpty ? 'empty' : '${b[0]}:${b[b.length - 1]}:${b.length}')
      .toSet()
      .toList();
}

@pragma('vm:entry-point')
void main() {
  assert(collectNarrowDigitBarcodes(['111', '121'], 1).toString() == '[1:1:3]');
  assert(collectNarrowDigitBarcodes(['', ''], 0).length == 1);
  assert(collectNarrowDigitBarcodes(['12', '21', '12'], 2).toString() == '[1:2:2, 2:1:2]');
  print('All tests passed!');
}