@pragma('vm:entry-point')
bool acceptsPacketSizeBurst(String stream, int maxSize) {
  int value = -1;
  for (int i = 0; i < stream.length; i++) {
    int c = stream.codeUnitAt(i);
    if (c >= 48 && c <= 57) {
      value = (value < 0 ? 0 : value) * 10 + c - 48;
    } else if (c == 124) {
      if (value <= 0 || value > maxSize) return false;
      value = -1;
    } else {
      return i == stream.length - 1 && c == 33 && value > 0 && value <= maxSize;
    }
  }
  return false;
}

@pragma('vm:entry-point')
void main() {
  assert(acceptsPacketSizeBurst('12|7!', 20) == true);
  assert(acceptsPacketSizeBurst('12|21!', 20) == false);
  assert(acceptsPacketSizeBurst('5|0!', 10) == false);
  print('All tests passed!');
}