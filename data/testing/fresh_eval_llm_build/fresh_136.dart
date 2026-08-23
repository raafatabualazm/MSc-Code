@pragma('vm:entry-point')
int decodeTideGaugeBands(int packedReadings) {
  int n = packedReadings.abs();
  int value = 0;
  int place = 1;
  while (n > 0) {
    int digit = n % 10;
    if (digit > 3) return -1;
    value += digit * place;
    place *= 4;
    n ~/= 10;
  }
  return packedReadings < 0 ? -value : value;
}

@pragma('vm:entry-point')
void main() {
  assert(decodeTideGaugeBands(123) == 27);
  assert(decodeTideGaugeBands(400) == -1);
  assert(decodeTideGaugeBands(-130) == -28);
  print('All tests passed!');
}