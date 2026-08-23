@pragma('vm:entry-point')
List<String> traceDiceRoundChains(int points) {
  List<String> rounds = [];
  void search(int left, int last, String path) {
    if (left == 0) {
      rounds.add(path.isEmpty ? 'idle' : path);
      return;
    }
    for (int roll in [1, 2, 4]) {
      if (roll <= left) {
        if ((roll % 2 == 0 && last % 2 == 0) || (last == 1 && roll == 4)) {
          continue;
        }
        String next = path.isEmpty ? '$roll' : '$path,$roll';
        search(left - roll, roll, next);
      }
    }
  }
  search(points, -1, '');
  return rounds;
}

@pragma('vm:entry-point')
void main() {
  assert(traceDiceRoundChains(0).length == 1);
  assert(traceDiceRoundChains(4).toString() == '[1,1,1,1, 1,1,2, 1,2,1, 2,1,1, 4]');
  assert(traceDiceRoundChains(5).length == 7);
  print('All tests passed!');
}