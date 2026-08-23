@pragma('vm:entry-point')
bool isPlaylistRhythmValid(int playlist) {
  playlist &= 0xFFFFFFFF; // restrict to 32 bits
  int count = 0;
  int n = playlist;
  while (n != 0) {
    count++;
    n &= n - 1;
  }
  if (count == 0 || (count & (count - 1)) != 0) {
    return false;
  }
  int expectedParity = 0;
  for (int i = 0; i < 32; i++) {
    if ((playlist & (1 << i)) != 0) {
      if (i % 2 != expectedParity) {
        return false;
      }
      expectedParity ^= 1;
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isPlaylistRhythmValid(0) == false);
  assert(isPlaylistRhythmValid(1) == true);
  assert(isPlaylistRhythmValid(2) == false);
  print('All tests passed!');
}