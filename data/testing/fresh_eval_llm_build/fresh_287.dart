@pragma('vm:entry-point')
int manifestDockConflictScore(List<int> manifest) {
  if (manifest.isEmpty) return 0;
  int left = manifest[0] > manifest[4] ? manifest[0] : manifest[4];
  int right = (manifest[0] + manifest[2]) < (manifest[4] + manifest[6]) ? (manifest[0] + manifest[2]) : (manifest[4] + manifest[6]);
  int bottom = manifest[1] > manifest[5] ? manifest[1] : manifest[5];
  int top = (manifest[1] + manifest[3]) < (manifest[5] + manifest[7]) ? (manifest[1] + manifest[3]) : (manifest[5] + manifest[7]);
  if (left < right && bottom < top) return (right - left) * (top - bottom);
  return (right < left ? left - right : 0) + (top < bottom ? bottom - top : 0);
}

@pragma('vm:entry-point')
void main() {
  assert(manifestDockConflictScore([]) == 0);
  assert(manifestDockConflictScore([0, 0, 4, 3, 2, 1, 3, 3]) == 4);
  assert(manifestDockConflictScore([0, 0, 2, 2, 5, 4, 1, 1]) == 5);
  print('All tests passed!');
}