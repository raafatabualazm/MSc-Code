@pragma('vm:entry-point')
bool isValidSoilMoistureGrid(String encodedGrid) {
  if (encodedGrid.isEmpty) return false;
  var rows = encodedGrid.split('|');
  int? rowLength;
  for (var row in rows) {
    if (row.isEmpty) return false;
    int hyphenIdx = row.lastIndexOf('-');
    if (hyphenIdx == -1 || hyphenIdx != row.length - 2) return false;
    String checksumChar = row[hyphenIdx + 1];
    if (checksumChar.codeUnitAt(0) < 48 || checksumChar.codeUnitAt(0) > 57) return false;
    int providedChecksum = int.parse(checksumChar);
    String encoded = row.substring(0, hyphenIdx);
    if (encoded.isEmpty || encoded.length % 2 != 0) return false;
    int sum = 0;
    int decodedLen = 0;
    for (int i = 0; i < encoded.length; i += 2) {
      String cChar = encoded[i];
      String dChar = encoded[i + 1];
      if (cChar.codeUnitAt(0) < 49 || cChar.codeUnitAt(0) > 57) return false;
      if (dChar.codeUnitAt(0) < 48 || dChar.codeUnitAt(0) > 57) return false;
      int count = int.parse(cChar);
      int digit = int.parse(dChar);
      sum += count * digit;
      decodedLen += count;
    }
    if (sum % 10 != providedChecksum) return false;
    if (rowLength == null) {
      rowLength = decodedLen;
    } else if (decodedLen != rowLength) {
      return false;
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isValidSoilMoistureGrid('') == false);
  assert(isValidSoilMoistureGrid('1521-7') == true);
  assert(isValidSoilMoistureGrid('1521-7|2312-8') == true);
  print('All tests passed!');
}