@pragma('vm:entry-point')
int decodeRleTimetableChecksum(String encoded) {
  if (encoded.isEmpty) return 0;
  final parts = encoded.split(',');
  int total = 0;
  for (final part in parts) {
    final sides = part.split('x');
    final count = int.parse(sides[0]);
    final minute = int.parse(sides[1]);
    total += count * minute;
  }
  return total - parts.length;
}

@pragma('vm:entry-point')
void main() {
  assert(decodeRleTimetableChecksum('') == 0);
  assert(decodeRleTimetableChecksum('2x15,3x30,1x45') == 162);
  assert(decodeRleTimetableChecksum('1x10') == 9);
  print('All tests passed!');
}