@pragma('vm:entry-point')
String describeMazeCellJoints(int cell) {
  int mask = cell & 15;
  int rot = 0;
  while (rot < 4 && (mask & 1) == 0) {
    mask = ((mask >> 1) | ((mask & 1) << 3)) & 15;
    rot++;
  }
  int bits = (mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1) + ((mask >> 3) & 1);
  if (bits == 0) return 'sealed';
  if (bits == 4) return 'hub';
  return bits == 2 && mask == 5 ? 'hall:$rot' : bits == 2 ? 'bend:$rot' : bits == 1 ? 'spur:$rot' : 'fork:$rot';
}

@pragma('vm:entry-point')
void main() {
  assert(describeMazeCellJoints(0) == 'sealed');
  assert(describeMazeCellJoints(10) == 'hall:1');
  assert(describeMazeCellJoints(14) == 'fork:1');
  print('All tests passed!');
}