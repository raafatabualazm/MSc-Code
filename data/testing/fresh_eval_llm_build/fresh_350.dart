@pragma('vm:entry-point')
Map<String, int> findThresholdRound(List<int> cumulativeVotes, int target) {
  if (cumulativeVotes.isEmpty) return {'round': 0, 'votes': 0};
  if (target > cumulativeVotes.last) return {'round': cumulativeVotes.length + 1, 'votes': cumulativeVotes.last};
  int low = 0, high = cumulativeVotes.length - 1;
  while (low < high) {
    int mid = (low + high) ~/ 2;
    if (cumulativeVotes[mid] < target) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return {'round': low + 1, 'votes': cumulativeVotes[low]};
}

@pragma('vm:entry-point')
void main() {
  assert(findThresholdRound([10, 20, 30], 25)['round'] == 3);
  assert(findThresholdRound([5, 10, 15], 5)['round'] == 1);
  assert(findThresholdRound([], 10)['round'] == 0);
  print('All tests passed!');
}