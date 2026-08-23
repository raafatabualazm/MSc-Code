@pragma('vm:entry-point')
int countPlaylistLoopSwitches(int playlistMask) {
  int switches = 0;
  for (int i = 0; i < 12; i++) {
    int current = (playlistMask >> i) & 1;
    int next = (playlistMask >> ((i + 1) % 12)) & 1;
    if (current != next) {
      switches++;
    }
  }
  return switches;
}

@pragma('vm:entry-point')
void main() {
  assert(countPlaylistLoopSwitches(0) == 0);
  assert(countPlaylistLoopSwitches(1) == 2);
  assert(countPlaylistLoopSwitches(1365) == 12);
  print('All tests passed!');
}