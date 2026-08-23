@pragma('vm:entry-point')
List<int> traceBracketChaos(List<int> seeds) {
  if (seeds.isEmpty) return [];
  List<int> chaos = [];
  int size = 1;
  while (size < seeds.length) {
    int grow = 1;
    while (grow < size) {
      grow *= 2;
    }
    size += grow;
  }
  int play(int start, int end, int depth) {
    if (end - start == 1) {
      return start < seeds.length ? seeds[start] : 0;
    }
    int mid = (start + end) ~/ 2;
    int left = play(start, mid, depth + 1);
    int right = play(mid, end, depth + 1);
    while (chaos.length <= depth) {
      chaos.add(0);
    }
    if (left <= 0 && right <= 0) return 0;
    if (left <= 0 || right <= 0) {
      chaos[depth] += 1;
      return left > right ? left : right;
    }
    int winner;
    if ((left + right) % 3 == 0) {
      winner = left > right ? left : right;
      chaos[depth] += 2;
    } else {
      winner = left < right ? left : right;
      if ((left - right).abs() == 1) {
        chaos[depth] += 1;
      }
    }
    return winner;
  }
  play(0, size, 0);
  for (int i = chaos.length - 1; i >= 0; i--) {
    if (chaos[i] != 0) break;
    chaos.removeLast();
  }
  return chaos;
}

@pragma('vm:entry-point')
void main() {
  assert(traceBracketChaos([1, 2]).toString() == '[2]');
  assert(traceBracketChaos([1, 3]).toString() == '[]');
  assert(traceBracketChaos([1, 2, 3, 4]).toString() == '[1, 3]');
  print('All tests passed!');
}