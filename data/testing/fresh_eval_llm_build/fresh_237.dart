@pragma('vm:entry-point')
List<int> collectAlignedPacketSizes(String packetStream, int alignmentSize) {
  List<int> result = [];
  for (String token in packetStream.split('|')) {
    if (token.endsWith('B')) {
      int size = int.parse(token.substring(0, token.length - 1));
      if (size > 0 && size % alignmentSize == 0) result.add(size);
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(collectAlignedPacketSizes("64B|65B|128B", 64).toString() == "[64, 128]");
  assert(collectAlignedPacketSizes("", 8).toString() == "[]");
  assert(collectAlignedPacketSizes("-32B|32B|96B", 32).length == 2);
  print('All tests passed!');
}