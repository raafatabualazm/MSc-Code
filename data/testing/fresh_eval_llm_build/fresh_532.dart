@pragma('vm:entry-point')
int countPrimeGreenSeconds(String phases) {
  int total = 0, i = 0, len = phases.length;
  while (i < len) {
    String letter = phases[i++];
    int duration = 0;
    while (i < len) {
      int? digit = int.tryParse(phases[i]);
      if (digit == null) break;
      duration = duration * 10 + digit;
      i++;
    }
    if (letter == 'G') {
      for (int sec = 1; sec <= duration; sec++) {
        if (sec < 2) continue;
        bool isPrime = true;
        for (int j = 2; j * j <= sec; j++) {
          if (sec % j == 0) {
            isPrime = false;
            break;
          }
        }
        if (isPrime) total++;
      }
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(countPrimeGreenSeconds("R3G5Y2") == 3);
  assert(countPrimeGreenSeconds("") == 0);
  assert(countPrimeGreenSeconds("G11") == 5);
  print('All tests passed!');
}