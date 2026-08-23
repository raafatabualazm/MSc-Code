@pragma('vm:entry-point')
bool isValidEditEncoding(String encoded) {
  if (encoded.isEmpty) return false;
  int i = 0;
  bool expectingLetter = true;
  int totalSum = 0;
  int sumI = 0;
  int sumD = 0;
  String? currentLetter;
  while (i < encoded.length) {
    if (expectingLetter) {
      int code = encoded.codeUnitAt(i);
      if (code != 82 && code != 73 && code != 68) return false;
      currentLetter = encoded[i];
      expectingLetter = false;
      i++;
    } else {
      int code = encoded.codeUnitAt(i);
      if (code < 48 || code > 57) return false;
      int j = i;
      while (j < encoded.length) {
        int c = encoded.codeUnitAt(j);
        if (c < 48 || c > 57) break;
        j++;
      }
      String numStr = encoded.substring(i, j);
      if (numStr[0] == '0') return false;
      int count = int.parse(numStr);
      totalSum += count;
      if (totalSum > 15) return false;
      if (currentLetter == 'I') {
        sumI += count;
      } else if (currentLetter == 'D') {
        sumD += count;
      }
      i = j;
      expectingLetter = true;
    }
  }
  return expectingLetter && sumI <= sumD;
}

@pragma('vm:entry-point')
void main() {
  assert(isValidEditEncoding('R1') == true);
  assert(isValidEditEncoding('I1') == false);
  assert(isValidEditEncoding('') == false);
  print('All tests passed!');
}