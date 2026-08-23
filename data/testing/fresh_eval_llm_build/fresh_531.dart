@pragma('vm:entry-point')
bool validateElevatorRequestTape(String tape) {
  int i = 0;
  int floor = 0;
  while (i < tape.length) {
    if (tape[i] != 'E') return false;
    i++;
    bool moved = false;
    bool canceled = false;
    while (i < tape.length && tape[i] != '!') {
      String c = tape[i];
      if (c == 'x') {
        if (!moved || canceled) return false;
        canceled = true;
        i++;
        continue;
      }
      if ((c != 'u' && c != 'd') || i + 1 >= tape.length) return false;
      int step = tape.codeUnitAt(i + 1) - 48;
      if (step < 1 || step > 3) return false;
      for (int k = 0; k < step; k++) {
        floor += c == 'u' ? 1 : -1;
        if (floor < -2 || floor > 9) return false;
      }
      moved = true;
      i += 2;
    }
    if (!moved || i >= tape.length || tape[i] != '!') return false;
    if (canceled && floor == 0) return false;
    i++;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(validateElevatorRequestTape('Eu1!') == true);
  assert(validateElevatorRequestTape('E!') == false);
  assert(validateElevatorRequestTape('Eu3u3u3u1!') == false);
  print('All tests passed!');
}