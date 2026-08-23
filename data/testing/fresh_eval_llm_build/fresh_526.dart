@pragma('vm:entry-point')
int thermostatScheduleChecksum(String encoded, int startTemp) {
  int temp = startTemp;
  int score = 0;
  int i = 0;
  while (i < encoded.length) {
    String mode = encoded[i];
    if (mode != 'H' && mode != 'C' && mode != 'S') {
      return -1;
    }
    i++;
    int count = 0;
    while (i < encoded.length) {
      int digit = encoded.codeUnitAt(i) - 48;
      if (digit < 0 || digit > 9) {
        break;
      }
      count = count * 10 + digit;
      i++;
    }
    if (count == 0) {
      continue;
    }
    for (int step = 0; step < count; step++) {
      if (mode == 'H') {
        temp += 2;
      } else if (mode == 'C') {
        temp -= 3;
      } else {
        if (temp > startTemp) {
          temp--;
        } else if (temp < startTemp) {
          temp++;
        }
      }
      score += (temp.abs() + step + count) % 7;
      if (temp >= startTemp + 6) {
        score += 4;
      } else if (temp <= startTemp - 6) {
        score -= 5;
      }
    }
  }
  return score + temp;
}

@pragma('vm:entry-point')
void main() {
  assert(thermostatScheduleChecksum('', 5) == 5);
  assert(thermostatScheduleChecksum('H3', 0) == 20);
  assert(thermostatScheduleChecksum('X2', 4) == -1);
  print('All tests passed!');
}