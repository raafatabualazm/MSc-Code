@pragma('vm:entry-point')
String playlistFeatureSummary(int flags) {
  int masked = flags & 0xF;
  int count = 0;
  int tmp = masked;
  while (tmp != 0) {
    count += tmp & 1;
    tmp >>= 1;
  }
  if (count == 0) return 'silent';
  if (count <= 2) return 'basic:$count';
  return 'rich:$count';
}

@pragma('vm:entry-point')
void main() {
  assert(playlistFeatureSummary(0) == 'silent');
  assert(playlistFeatureSummary(7) == 'rich:3');
  assert(playlistFeatureSummary(3) == 'basic:2');
  print('All tests passed!');
}