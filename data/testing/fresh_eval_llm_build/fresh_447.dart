@pragma('vm:entry-point')
int playlistRevisitBalance(List<int> releaseDays) {
  if (releaseDays.length < 2) return 0;
  int score = 0;
  for (int i = 1; i < releaseDays.length; i++) {
    int gap = releaseDays[i] - releaseDays[i - 1];
    if (gap < 0) {
      score -= 6;
    } else if (gap == 0) {
      score -= 3;
    } else if (gap <= 2) {
      score += 4;
    } else if (gap <= 6) {
      score += 1;
    } else {
      score -= gap - 6;
    }
    if (i % 2 == 0) {
      score += gap.isOdd ? 2 : -1;
    }
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(playlistRevisitBalance([1, 2, 5]) == 7);
  assert(playlistRevisitBalance([0, 7]) == -1);
  assert(playlistRevisitBalance([1, 1, 2]) == 3);
  print('All tests passed!');
}