@pragma('vm:entry-point')
String buildPacketTree(int dataSize, int maxChunk) {
  if (dataSize <= maxChunk) {
    return '$dataSize';
  }
  int left = dataSize ~/ 2;
  return '(${buildPacketTree(left, maxChunk)},${buildPacketTree(dataSize - left, maxChunk)})';
}

@pragma('vm:entry-point')
void main() {
  assert(buildPacketTree(0, 100) == '0');
  assert(buildPacketTree(5, 2) == '(2,(1,2))');
  assert(buildPacketTree(8, 3) == '((2,2),(2,2))');
  print('All tests passed!');
}