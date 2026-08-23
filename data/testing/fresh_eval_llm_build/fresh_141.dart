@pragma('vm:entry-point')
int evenRunBucketChecksum(String encoded) {
  int total = 0;
  int i = 0;
  while (i + 1 < encoded.length) {
    int count = int.parse(encoded[i]);
    int charCode = encoded.codeUnitAt(i + 1);
    if (count % 2 == 0) {
      total += count * charCode;
    }
    i += 2;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(evenRunBucketChecksum("") == 0);
  assert(evenRunBucketChecksum("2a") == 194);
  assert(evenRunBucketChecksum("2a4b6c") == 1180);
  print('All tests passed!');
}