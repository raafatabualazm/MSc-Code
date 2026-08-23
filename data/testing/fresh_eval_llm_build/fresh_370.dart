@pragma('vm:entry-point')
List<int> countPhaseCategories(List<String> phases) {
  const durationMap = {
    'red': 30,
    'yellow': 3,
    'green': 30,
    'arrow': 10,
    'allRed': 60,
  };
  int short = 0, medium = 0, long = 0;
  for (var p in phases) {
    int d = durationMap[p] ?? 0;
    if (d < 10) {
      short++;
    } else if (d <= 30) {
      medium++;
    } else {
      long++;
    }
  }
  return [short, medium, long];
}

@pragma('vm:entry-point')
void main() {
  assert(countPhaseCategories([]).toString() == '[0, 0, 0]');
  assert(countPhaseCategories(['yellow', 'red', 'allRed']).toString() == '[1, 1, 1]');
  assert(countPhaseCategories(['arrow']).toString() == '[0, 1, 0]');
  print('All tests passed!');
}