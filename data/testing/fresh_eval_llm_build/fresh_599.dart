@pragma('vm:entry-point')
int spellPatchOverlapScore(List<String> edits) {
  List<List<int>> marks = [];
  int total = 0;
  for (int i = 0; i < edits.length; i++) {
    int x = 0, y = 0, minX = 0, maxX = 0, minY = 0, maxY = 0, seen = 0;
    for (int j = 0; j < edits[i].length; j++) {
      String ch = edits[i][j];
      if (ch == 'L') x--;
      else if (ch == 'R') x++;
      else if (ch == 'U') y++;
      else if (ch == 'D') y--;
      else if (ch != '!') continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      if (ch == '!') {
        marks.add([x, y, i]);
        seen++;
      }
    }
    if (seen == 0) continue;
    int area = (maxX - minX) * (maxY - minY);
    total += area == 0 ? seen : area + seen;
  }
  for (int i = 0; i < marks.length; i++) {
    for (int j = i + 1; j < marks.length; j++) {
      if (marks[i][2] == marks[j][2]) continue;
      int d = (marks[i][0] - marks[j][0]).abs() + (marks[i][1] - marks[j][1]).abs();
      if (d == 0) total += 4;
      else if (d == 1) total += 1;
      else if (d == 2 && (marks[i][0] == marks[j][0] || marks[i][1] == marks[j][1])) total += 2;
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(spellPatchOverlapScore([]) == 0);
  assert(spellPatchOverlapScore(['R!', '!']) == 3);
  assert(spellPatchOverlapScore(['U!', 'D!', '!']) == 7);
  print('All tests passed!');
}