@pragma('vm:entry-point')
int evaluatePacketBurstMatrix(List<List<int>> packets) {
  if (packets.isEmpty) return 0;
  int score = 0;
  for (int r = 0; r < packets.length; r++) {
    if (packets[r].isEmpty) continue;
    bool rowBurst = false;
    for (int c = 0; c < packets[r].length; c++) {
      int v = packets[r][c];
      if (v == 0) continue;
      if (v < 0) {
        score -= (r == 0 || c == 0 || c == packets[r].length - 1) ? v.abs() : 2;
        continue;
      }
      score += v % 10;
      if (c > 0 && packets[r][c - 1] == v) {
        score += 3;
        rowBurst = true;
      } else if (c > 0 && packets[r][c - 1] > v) {
        score -= 1;
      }
      if (r > 0 && c < packets[r - 1].length) {
        int above = packets[r - 1][c];
        if (v > above) {
          score += 2;
        } else if (v < above - 4) {
          score -= 2;
        }
      }
      if (r > 0 && c > 0 && ((v + packets[r - 1][c - 1]) % 5 == 0)) {
        score += 1;
      }
    }
    if (rowBurst) score += 4;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(evaluatePacketBurstMatrix([[5]]) == 5);
  assert(evaluatePacketBurstMatrix([[5, 5]]) == 17);
  assert(evaluatePacketBurstMatrix([[7, 1], [1, 7]]) == 15);
  print('All tests passed!');
}