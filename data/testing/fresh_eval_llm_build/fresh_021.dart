@pragma('vm:entry-point')
String packetSizeBucketSummary(List<int> sizes) {
  int small = 0, medium = 0, large = 0;
  for (var size in sizes) {
    if (size <= 512) {
      small++;
    } else if (size <= 1024) {
      medium++;
    } else {
      large++;
    }
  }
  return "small:$small medium:$medium large:$large";
}

@pragma('vm:entry-point')
void main() {
  assert(packetSizeBucketSummary([]) == "small:0 medium:0 large:0");
  assert(packetSizeBucketSummary([512, 513, 1024, 1025]) == "small:1 medium:2 large:1");
  assert(packetSizeBucketSummary([100, 200, 300]) == "small:3 medium:0 large:0");
  print('All tests passed!');
}