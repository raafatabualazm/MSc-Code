@pragma('vm:entry-point')
List<int> computeShelfGroupKeys(List<String> shelfCodes) {
  var result = <int>[];
  for (var code in shelfCodes) {
    if (code.isEmpty) {
      result.add(0);
      continue;
    }
    int letters = 0, digits = 0;
    bool hasVowel = false;
    bool zFound = false;
    for (var i = 0; i < code.length; i++) {
      var ch = code[i];
      if (ch == 'Z') {
        zFound = true;
        break;
      }
      if (ch.codeUnitAt(0) >= 65 && ch.codeUnitAt(0) <= 90) {
        letters++;
        if ('AEIOU'.contains(ch)) hasVowel = true;
      } else if (ch.codeUnitAt(0) >= 48 && ch.codeUnitAt(0) <= 57) {
        digits += int.parse(ch);
      }
    }
    if (zFound) {
      result.add(9999);
      continue;
    }
    int score;
    if (hasVowel) {
      if (digits > 9) {
        score = letters * 10 + digits % 10;
      } else {
        score = letters * 100 + digits;
      }
    } else {
      if (letters > 5) {
        score = letters * 2 + digits;
      } else {
        score = letters * 5 + digits;
      }
    }
    result.add(score);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(computeShelfGroupKeys(["A1"])[0] == 101);
  assert(computeShelfGroupKeys([]).isEmpty);
  assert(computeShelfGroupKeys(["Z"]).toString() == "[9999]");
  print('All tests passed!');
}