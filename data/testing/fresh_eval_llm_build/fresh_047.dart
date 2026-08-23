@pragma('vm:entry-point')
bool hasAuditPartition(List<int> ledger) {
  if (ledger.isEmpty) return false;
  int totalAbs = 0;
  for (int e in ledger) totalAbs += e.abs();
  if (totalAbs == 0 || totalAbs % 2 != 0) return false;
  int target = totalAbs ~/ 2;
  int n = ledger.length;
  // Backtracking: find subset with absSum==target, hasPos, hasNeg
  bool backtrack(int idx, int remaining, bool hasPos, bool hasNeg) {
    if (remaining == 0) return hasPos && hasNeg;
    if (idx >= n || remaining < 0) return false;
    for (int i = idx; i < n; i++) {
      int val = ledger[i];
      int av = val.abs();
      if (av > remaining) continue;
      bool np = hasPos || val > 0;
      bool nn = hasNeg || val < 0;
      if (backtrack(i + 1, remaining - av, np, nn)) return true;
      // skip duplicates by abs value to prune
      int j = i + 1;
      while (j < n && ledger[j].abs() == av && ledger[j].sign == val.sign) j++;
      i = j - 1;
    }
    return false;
  }
  return backtrack(0, target, false, false);
}

@pragma('vm:entry-point')
void main() {
  assert(hasAuditPartition([3, -2, 2, -1]) == true);
  assert(hasAuditPartition([1, 2, -3]) == false);
  assert(hasAuditPartition([]) == false);
  print('All tests passed!');
}