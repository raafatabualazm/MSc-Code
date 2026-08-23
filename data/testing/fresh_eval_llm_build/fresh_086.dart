@pragma('vm:entry-point')
bool matchesShelfLocatorTag(String code) {
  var parts = code.split('|');
  if (parts.length != 3 || parts[0].isEmpty || parts[1].length != 2) {
    return false;
  }
  var d1 = parts[1].codeUnitAt(0) - 48;
  var d2 = parts[1].codeUnitAt(1) - 48;
  return d1 >= 0 && d1 <= 9 &&
      d2 >= 0 && d2 <= 9 &&
      parts[2] == '${parts[0][0]}${d1 + d2}';
}

@pragma('vm:entry-point')
void main() {
  assert(matchesShelfLocatorTag('History|34|H7') == true);
  assert(matchesShelfLocatorTag('Art|19|A9') == false);
  assert(matchesShelfLocatorTag('|34|H7') == false);
  print('All tests passed!');
}