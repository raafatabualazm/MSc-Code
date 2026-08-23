@pragma('vm:entry-point')
int sampleResidueBalance(List<int> samples, int modulus, int threshold) {
  if (samples.isEmpty) return 0;
  int above = 0, below = 0;
  for (int sample in samples) {
    int r = sample % modulus;
    if (sample > threshold) {
      above += r;
    } else {
      below += r;
    }
  }
  return (above - below).abs();
}

@pragma('vm:entry-point')
void main() {
  assert(sampleResidueBalance([10, 20, 30], 5, 15) == 0);
  assert(sampleResidueBalance([1, 2, 3, 4, 5], 3, 2) == 0);
  assert(sampleResidueBalance([7, 8, 9], 4, 6) == 4);
  print('All tests passed!');
}