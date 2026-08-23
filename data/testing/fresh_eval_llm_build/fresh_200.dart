@pragma('vm:entry-point')
int countSignalDropTransitions(String readings) {
  int count = 0;
  bool inStrongRun = false;
  for (int i = 0; i < readings.length; i++) {
    final c = readings[i];
    if (c == 'A' || c == 'B' || c == 'C') {
      inStrongRun = true;
    } else if (c == 'D' || c == 'E' || c == 'F') {
      if (inStrongRun) {
        count++;
        inStrongRun = false;
      }
    } else {
      inStrongRun = false;
    }
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countSignalDropTransitions('') == 0);
  assert(countSignalDropTransitions('ACBDABEF') == 2);
  assert(countSignalDropTransitions('ABCFABD') == 2);
  print('All tests passed!');
}