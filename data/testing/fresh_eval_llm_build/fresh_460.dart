@pragma('vm:entry-point')
bool hasMatchingDnaTerritories(List<String> strands) {
  if (strands.length < 2) return false;
  List<List<int>> boxes = [];
  for (String s in strands) {
    int x = 0, y = 0, minX = 0, maxX = 0, minY = 0, maxY = 0;
    for (int i = 0; i < s.length; i++) {
      String c = s[i];
      if (c == 'A') {
        y++;
      } else if (c == 'T') {
        y--;
      } else if (c == 'C') {
        x--;
      } else if (c == 'G') {
        x++;
      } else {
        return false;
      }
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    boxes.add([minX, maxX, minY, maxY, x.abs() + y.abs()]);
  }
  for (int i = 0; i < boxes.length; i++) {
    for (int j = i + 1; j < boxes.length; j++) {
      if (boxes[i][4] != boxes[j][4] || boxes[i][4] == 0) continue;
      int overlapW = (boxes[i][1] < boxes[j][1] ? boxes[i][1] : boxes[j][1]) - (boxes[i][0] > boxes[j][0] ? boxes[i][0] : boxes[j][0]);
      int overlapH = (boxes[i][3] < boxes[j][3] ? boxes[i][3] : boxes[j][3]) - (boxes[i][2] > boxes[j][2] ? boxes[i][2] : boxes[j][2]);
      if (overlapW > 0 && overlapH > 0) return true;
    }
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(hasMatchingDnaTerritories(['AG', 'GA']) == true);
  assert(hasMatchingDnaTerritories(['A', 'T']) == false);
  assert(hasMatchingDnaTerritories(['AGX', 'GA']) == false);
  print('All tests passed!');
}