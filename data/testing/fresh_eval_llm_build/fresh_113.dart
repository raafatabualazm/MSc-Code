@pragma('vm:entry-point')
int countSamplesOverNoiseFloor(String rleData, int threshold) {
  int total = 0;
  for (var segment in rleData.split(' ')) {
    if (segment.isEmpty) continue;
    var parts = segment.split(':');
    int count = int.parse(parts[0]);
    int value = int.parse(parts[1]);
    if (value > threshold) {
      total += count;
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(countSamplesOverNoiseFloor("1:10", 5) == 1);
  assert(countSamplesOverNoiseFloor("", 0) == 0);
  assert(countSamplesOverNoiseFloor("2:3 3:7", 4) == 3);
  print('All tests passed!');
}