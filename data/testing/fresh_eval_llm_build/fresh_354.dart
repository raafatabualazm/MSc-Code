@pragma('vm:entry-point')
List<int> bucketPacketSizeRotations(List<int> sizes) {
  List<int> out = [0, 0, 0, 0];
  for (int size in sizes) {
    int byte = size & 255;
    int rot = ((byte << 1) & 255) | (byte >> 7);
    int bits = 0;
    int v = rot;
    while (v != 0) {
      bits++;
      v &= v - 1;
    }
    if ((bits & 1) == 0) {
      if ((rot & 24) == 24) {
        out[0]++;
      } else {
        out[1] += rot & 3;
      }
    } else if ((rot & 3) == 0) {
      out[2] += bits;
    } else {
      out[3] += size & 7;
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(bucketPacketSizeRotations([]).toString() == '[0, 0, 0, 0]');
  assert(bucketPacketSizeRotations([1, 2, 3]).toString() == '[0, 2, 1, 1]');
  assert(bucketPacketSizeRotations([12, 13, 14, 15]).toString() == '[2, 0, 3, 5]');
  print('All tests passed!');
}