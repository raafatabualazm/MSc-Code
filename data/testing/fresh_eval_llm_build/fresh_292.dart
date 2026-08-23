@pragma('vm:entry-point')
String? firstMutedRgbToken(String pixels, int maxChannelSum) {
  for (final token in pixels.split(' ')) {
    if (token.isEmpty) continue;
    final parts = token.split(',');
    final sum = int.parse(parts[0]) + int.parse(parts[1]) + int.parse(parts[2]);
    if (sum <= maxChannelSum) return token;
  }
  return null;
}

@pragma('vm:entry-point')
void main() {
  assert(firstMutedRgbToken('255,0,0 1,1,1', 3) == '1,1,1');
  assert(firstMutedRgbToken('', 5) == null);
  assert(firstMutedRgbToken('2,2,2 3,3,3', 5) == null);
  print('All tests passed!');
}