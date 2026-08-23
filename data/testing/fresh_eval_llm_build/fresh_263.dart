@pragma('vm:entry-point')
Map<String, int> classifyPacketSizesByUniqueness(List<int> packetSizes) {
  Set<int> seen = <int>{};
  Set<int> multiples = <int>{};
  for (int size in packetSizes) {
    if (seen.contains(size)) {
      if (!multiples.contains(size)) {
        multiples.add(size);
      }
    } else {
      seen.add(size);
    }
  }
  return {
    'unique': seen.length - multiples.length,
    'duplicate': multiples.length
  };
}

@pragma('vm:entry-point')
void main() {
  assert(classifyPacketSizesByUniqueness([])["unique"] == 0);
  assert(classifyPacketSizesByUniqueness([7])["unique"] == 1);
  assert(classifyPacketSizesByUniqueness([3,3])["duplicate"] == 1);
  print('All tests passed!');
}