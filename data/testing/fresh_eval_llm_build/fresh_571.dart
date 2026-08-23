@pragma('vm:entry-point')
bool verifiesElectionAuditIntervals(List<int> ledger) {
  List<int> checkpoints = [];
  for (int i = 0; i < ledger.length; i++) {
    if (ledger[i] == 0) {
      checkpoints.add(i);
    }
  }
  if (checkpoints.length < 2) {
    return false;
  }
  for (int block = 0; block < checkpoints.length - 1; block++) {
    int start = checkpoints[block];
    int end = checkpoints[block + 1];
    int span = end - start;
    if ((block.isEven && span.isOdd) || (!block.isEven && span.isEven)) {
      return false;
    }
    int running = 0;
    bool touchedVote = false;
    List<int> seenTotals = [0];
    for (int day = start + 1; day < end; day++) {
      running += ledger[day];
      touchedVote = true;
      for (int k = 0; k < seenTotals.length; k++) {
        if (seenTotals[k] == running) {
          return false;
        }
      }
      if (running.abs() > span * 1000) {
        return false;
      }
      seenTotals.add(running);
    }
    if (!touchedVote || running == 0) {
      return false;
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(verifiesElectionAuditIntervals([0, 1, 2, 4, 0, 5, -1, 0]) == true);
  assert(verifiesElectionAuditIntervals([0, 1, 2, 0]) == false);
  assert(verifiesElectionAuditIntervals([0, 1, 0, 0]) == false);
  print('All tests passed!');
}