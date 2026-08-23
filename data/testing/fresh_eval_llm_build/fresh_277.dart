@pragma('vm:entry-point')
int peakRiserStackInChart(String chartLog) {
  int active = 0;
  int peak = 0;
  for (int i = 0; i < chartLog.length; i++) {
    if (chartLog[i] == 'R') {
      active++;
      if (active > peak) peak = active;
    } else if (chartLog[i] == 'F' && active > 0) {
      active--;
    }
  }
  return peak;
}

@pragma('vm:entry-point')
void main() {
  assert(peakRiserStackInChart("") == 0);
  assert(peakRiserStackInChart("RFRRFF") == 2);
  assert(peakRiserStackInChart("FFRRR") == 3);
  print('All tests passed!');
}