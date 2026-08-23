@pragma('vm:entry-point')
int countThermostatBounceUnits(List<List<int>> schedule) {
  int total = 0;
  for (final day in schedule) {
    int swing = day.isEmpty ? 0 : day.last - day.first;
    total += swing < 0 ? -swing * 2 : swing;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(countThermostatBounceUnits([]) == 0);
  assert(countThermostatBounceUnits([[68, 70, 72]]) == 4);
  assert(countThermostatBounceUnits([[75, 72], [60, 63]]) == 9);
  print('All tests passed!');
}