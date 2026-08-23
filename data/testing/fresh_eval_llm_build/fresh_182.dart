@pragma('vm:entry-point')
int calculateShelfRefileScore(List<String> codes) {
  final Set<String> seen = {};
  int score = 0;
  for (final code in codes) {
    if (code.isEmpty) {
      score -= 2;
      continue;
    }
    int digits = 0;
    int letters = 0;
    for (final unit in code.codeUnits) {
      if (unit >= 48 && unit <= 57) {
        digits++;
      } else if (unit >= 65 && unit <= 90) {
        letters++;
      }
    }
    if (seen.contains(code)) {
      score -= digits >= letters ? digits + 1 : letters;
    } else {
      seen.add(code);
      if (digits == letters) {
        score += code.length * 2;
      } else if (digits > letters) {
        score += digits - letters;
      } else {
        score += letters;
      }
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(calculateShelfRefileScore([]) == 0);
  assert(calculateShelfRefileScore(['A1', 'A1']) == 2);
  assert(calculateShelfRefileScore(['', 'ZZ99', 'ZZ99', '7']) == 4);
  print('All tests passed!');
}