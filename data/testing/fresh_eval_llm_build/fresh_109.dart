@pragma('vm:entry-point')
bool playlistUsesSingleMixes(int playlistMask) {
  if (playlistMask <= 0) return false;
  while (playlistMask > 0) {
    int pair = playlistMask & 3;
    if (pair == 3) return false;
    playlistMask >>= 2;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(playlistUsesSingleMixes(1) == true);
  assert(playlistUsesSingleMixes(3) == false);
  assert(playlistUsesSingleMixes(36) == true);
  print('All tests passed!');
}