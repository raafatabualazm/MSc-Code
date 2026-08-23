@pragma('vm:entry-point')
double peakQrModuleDensity(String modules) {
  int black = 0;
  double best = 0.0;
  for (int i = 0; i < modules.length; i++) {
    if (modules[i] == '#') black++;
    if (i >= 4 && modules[i - 4] == '#') black--;
    if (i >= 3 && black / 4.0 > best) best = black / 4.0;
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(peakQrModuleDensity('') == 0.0);
  assert(peakQrModuleDensity('####') == 1.0);
  assert(peakQrModuleDensity('#..#') == 0.5);
  print('All tests passed!');
}