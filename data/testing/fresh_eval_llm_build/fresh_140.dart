@pragma('vm:entry-point')
List<int> wifiSignalBinCounts(List<int> coords) {
  List<int> bins = [0, 0, 0, 0];
  for (int i = 0; i + 1 < coords.length; i += 2) {
    int dist = coords[i].abs() + coords[i + 1].abs();
    if (dist <= 3) bins[0]++;
    else if (dist <= 7) bins[1]++;
    else if (dist <= 14) bins[2]++;
    else bins[3]++;
  }
  return bins;
}

@pragma('vm:entry-point')
void main() {
  assert(wifiSignalBinCounts([]).toString() == '[0, 0, 0, 0]');
  assert(wifiSignalBinCounts([0, 0]).toString() == '[1, 0, 0, 0]');
  assert(wifiSignalBinCounts([1, 2, 4, 0, 10, -4, 20, 5]).toString() == '[1, 1, 1, 1]');
  print('All tests passed!');
}