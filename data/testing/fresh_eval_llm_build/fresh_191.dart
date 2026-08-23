@pragma('vm:entry-point')
String reorderLibraryShelfCodes(String shelfCodes) {
  if (shelfCodes.isEmpty) return '';
  var codes = shelfCodes.split('|');
  int digits(String s) => RegExp(r'\d').allMatches(s).length;
  codes.sort((a, b) {
    var diff = digits(a) - digits(b);
    return diff != 0 ? diff : (b.length != a.length ? b.length - a.length : a.compareTo(b));
  });
  return codes.join('|');
}

@pragma('vm:entry-point')
void main() {
  assert(reorderLibraryShelfCodes('') == '');
  assert(reorderLibraryShelfCodes('A1|LONG2|BB3') == 'LONG2|BB3|A1');
  assert(reorderLibraryShelfCodes('Z|AA|B1') == 'AA|Z|B1');
  print('All tests passed!');
}