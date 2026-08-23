@pragma('vm:entry-point')
int maximizeManifestTransferCredits(List<String> manifests) {
  if (manifests.isEmpty) return 0;
  int cellValue(String ch) {
    if (ch == '.') return 0;
    if (ch == 'S') return 4;
    if (ch == 'R') return -2;
    return ch.codeUnitAt(0) - 48;
  }

  List<List<int>> dp = manifests.map((r) => List.filled(r.length, -1000000000)).toList();
  for (int c = 0; c < manifests[0].length; c++) {
    if (manifests[0][c] != '#') dp[0][c] = cellValue(manifests[0][c]);
  }
  for (int r = 1; r < manifests.length; r++) {
    for (int c = 0; c < manifests[r].length; c++) {
      if (manifests[r][c] == '#') continue;
      int best = -1000000000;
      for (int pc = c - 1; pc <= c + 1; pc++) {
        if (pc < 0 || pc >= manifests[r - 1].length || dp[r - 1][pc] < -999999999) continue;
        int score = dp[r - 1][pc] + cellValue(manifests[r][c]);
        if (manifests[r - 1][pc] == manifests[r][c]) score -= 2;
        if (score > best) best = score;
      }
      dp[r][c] = best;
    }
  }
  int answer = -1000000000;
  for (int v in dp.last) {
    if (v > answer) answer = v;
  }
  return answer < -999999999 ? -1 : answer;
}

@pragma('vm:entry-point')
void main() {
  assert(maximizeManifestTransferCredits([]) == 0);
  assert(maximizeManifestTransferCredits(['12', '34']) == 6);
  assert(maximizeManifestTransferCredits(['1#2', '345', '67#']) == 14);
  print('All tests passed!');
}