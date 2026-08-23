@pragma('vm:entry-point')
int qrModuleChecksumDrift(String encoded) {
  if (encoded.isEmpty) return 0;
  int i = 0;
  int moduleIndex = 0;
  int checksum = 0;
  String expected = 'B';
  while (i < encoded.length) {
    String color = encoded[i];
    if (color == '|' || color == ' ') {
      i++;
      continue;
    }
    if (color != 'B' && color != 'W') return -1;
    i++;
    int count = 0;
    while (i < encoded.length) {
      int digit = encoded.codeUnitAt(i) - 48;
      if (digit < 0 || digit > 9) break;
      count = count * 10 + digit;
      i++;
    }
    if (count == 0) return -1;
    if (color != expected) checksum += 5;
    for (int j = 0; j < count; j++) {
      int pos = moduleIndex + j;
      if (color == 'B') {
        checksum += (pos % 3) + 1;
        if (pos.isEven && j > 0) checksum++;
      } else if (pos.isOdd) {
        checksum -= 2;
      } else {
        checksum--;
      }
    }
    moduleIndex += count;
    expected = color == 'B' ? 'W' : 'B';
  }
  return checksum + moduleIndex;
}

@pragma('vm:entry-point')
void main() {
  assert(qrModuleChecksumDrift('B1') == 2);
  assert(qrModuleChecksumDrift('B1W1') == 1);
  assert(qrModuleChecksumDrift('Q1') == -1);
  print('All tests passed!');
}