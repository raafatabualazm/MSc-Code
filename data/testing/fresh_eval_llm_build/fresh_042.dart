@pragma('vm:entry-point')
int finalVoteCountAfterChallenges(List<int> actions) {
  if (actions.isEmpty) return 0;
  List<int> votes = [];
  for (int action in actions) {
    if (action > 0) {
      votes.add(action);
    } else if (action == -1) {
      if (votes.isNotEmpty) {
        votes.removeLast();
      }
    } else if (action < -1) {
      int candidate = -action;
      for (int i = votes.length - 1; i >= 0; i--) {
        if (votes[i] == candidate) {
          votes.removeAt(i);
          break;
        }
      }
    }
  }
  return votes.length;
}

@pragma('vm:entry-point')
void main() {
  assert(finalVoteCountAfterChallenges([]) == 0);
  assert(finalVoteCountAfterChallenges([1, -1, 2]) == 1);
  assert(finalVoteCountAfterChallenges([3, 3, -3]) == 1);
  print('All tests passed!');
}