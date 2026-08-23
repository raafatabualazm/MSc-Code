@pragma('vm:entry-point')
int thermostatCycleChecksum(String encoded) {
  int total = 0;
  int i = 0;
  while (i < encoded.length) {
    String mode = encoded[i++];
    int minutes = 0;
    while (i < encoded.length &&
        encoded.codeUnitAt(i) >= 48 &&
        encoded.codeUnitAt(i) <= 57) {
      minutes = minutes * 10 + (encoded.codeUnitAt(i) - 48);
      i++;
    }
    if (mode == 'H') {
      total += minutes * 2;
      if (minutes >= 10) total += 5;
    } else if (mode == 'C') {
      total -= minutes * 3;
      if (minutes >= 10) total -= 5;
    } else {
      total += minutes.isEven ? minutes : -minutes;
      if (minutes == 0) total += 4;
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(thermostatCycleChecksum('H3C2') == 0);
  assert(thermostatCycleChecksum('E0') == 4);
  assert(thermostatCycleChecksum('H12E0C4') == 21);
  print('All tests passed!');
}