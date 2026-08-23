@pragma('vm:entry-point')
List<int> orderTournamentLadders(List<int> seeds, int upsetWindow) {
  if (seeds.isEmpty) {
    return [];
  }
  List<List<int>> ranked = [];
  for (int i = 0; i < seeds.length; i++) {
    int pressure = 0;
    for (int j = 0; j < seeds.length; j++) {
      if (i == j) {
        continue;
      }
      int gap = (seeds[i] - seeds[j]).abs();
      if (gap <= upsetWindow) {
        pressure += 3;
      } else if (seeds[i] < seeds[j]) {
        pressure += 1;
      } else {
        pressure -= 2;
      }
      if ((seeds[i] + seeds[j]) % 4 == 0) {
        pressure += 1;
      }
    }
    if (seeds[i] < 0) {
      pressure -= 4;
    } else if (seeds[i] == 0) {
      pressure += 2;
    }
    ranked.add([seeds[i], pressure]);
  }
  ranked.sort((a, b) {
    if (a[1] != b[1]) {
      return b[1].compareTo(a[1]);
    }
    bool aOdd = a[0].isOdd;
    bool bOdd = b[0].isOdd;
    if (aOdd != bOdd) {
      return aOdd ? -1 : 1;
    }
    int absCmp = a[0].abs().compareTo(b[0].abs());
    if (absCmp != 0) {
      return absCmp;
    }
    return a[0].compareTo(b[0]);
  });
  return ranked.map((e) => e[0]).toList();
}

@pragma('vm:entry-point')
void main() {
  assert(orderTournamentLadders([], 2).toString() == '[]');
  assert(orderTournamentLadders([3, 5, 8], 3).toString() == '[5, 3, 8]');
  assert(orderTournamentLadders([2, 5, 9], 3).toString() == '[5, 2, 9]');
  print('All tests passed!');
}