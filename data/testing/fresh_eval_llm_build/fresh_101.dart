@pragma('vm:entry-point')
num evaluateRgbTokenBalance(String stream, int limit) {
  if (stream.isEmpty) {
    return 0;
  }
  int score = 0;
  for (String token in stream.split('|')) {
    List<String> parts = token.split(':');
    int bright = 0;
    bool valid = parts.length == 3;
    for (String part in parts) {
      int? value = int.tryParse(part.trim());
      if (value == null || value < 0 || value > 255) {
        valid = false;
      } else if (value > limit) {
        bright++;
      }
    }
    if (!valid) {
      score -= 2;
    } else if (bright == 3) {
      score += 3;
    } else if (bright == 2) {
      score += 1;
    } else if (bright == 0) {
      score -= 1;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(evaluateRgbTokenBalance('', 10) == 0);
  assert(evaluateRgbTokenBalance('10:20:30', 15) == 1);
  assert(evaluateRgbTokenBalance('256:0:0|255:255:255', 200) == 1);
  print('All tests passed!');
}