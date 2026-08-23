@pragma('vm:entry-point')
int passwordPatternArea(String moves) {
  int x = 0, y = 0, minX = 0, maxX = 0, minY = 0, maxY = 0;
  for (int i = 0; i < moves.length; i++) {
    String c = moves[i];
    switch (c) {
      case 'U': y++; if (y > maxY) maxY = y; break;
      case 'D': y--; if (y < minY) minY = y; break;
      case 'L': x--; if (x < minX) minX = x; break;
      case 'R': x++; if (x > maxX) maxX = x; break;
    }
  }
  return (maxX - minX) * (maxY - minY);
}

@pragma('vm:entry-point')
void main() {
  assert(passwordPatternArea('') == 0);
  assert(passwordPatternArea('UR') == 1);
  assert(passwordPatternArea('LLDDR') == 4);
  print('All tests passed!');
}