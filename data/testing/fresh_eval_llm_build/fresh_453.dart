@pragma('vm:entry-point')
int evaluateWifiBinDrift(String scans, int allowedJump) {
  if (scans.isEmpty) return 0;
  int score = 0;
  for (String session in scans.split('|')) {
    int? previousBin;
    int validCount = 0;
    for (String raw in session.split(',')) {
      String token = raw.trim();
      if (token.isEmpty) continue;
      int? value = int.tryParse(token);
      if (value == null || value > 0 || value < -120) {
        score -= 2;
        continue;
      }
      int bin = value <= -80 ? 0 : (value <= -60 ? 1 : (value <= -40 ? 2 : 3));
      if (previousBin != null) {
        int jump = (bin - previousBin).abs();
        if (jump > allowedJump) {
          score += jump * 2;
        } else if (jump == 0) {
          score += 1;
        } else {
          score -= jump;
        }
      }
      previousBin = bin;
      validCount++;
    }
    if (validCount == 1) score += 3;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(evaluateWifiBinDrift("-80,-81,-82", 0) == 2);
  assert(evaluateWifiBinDrift("-50,abc,-90", 1) == 2);
  assert(evaluateWifiBinDrift("-80||-30,-30", 0) == 4);
  print('All tests passed!');
}