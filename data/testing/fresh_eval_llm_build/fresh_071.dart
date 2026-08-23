@pragma('vm:entry-point')
bool hasBalancedLedgerPostingOrder(List<String> entries) {
  List<String> ordered = List<String>.from(entries);
  ordered.sort((a, b) {
    List<String> pa = a.split(':');
    List<String> pb = b.split(':');
    int aa = int.parse(pa[1]).abs();
    int ab = int.parse(pb[1]).abs();
    if (aa != ab) return ab.compareTo(aa);
    if (pa[2] != pb[2]) return pa[2].compareTo(pb[2]);
    return pa[0].compareTo(pb[0]);
  });
  for (int i = 1; i < ordered.length; i++) {
    List<String> prev = ordered[i - 1].split(':');
    List<String> curr = ordered[i].split(':');
    int prevAbs = int.parse(prev[1]).abs();
    int currAbs = int.parse(curr[1]).abs();
    if (prev[0] == curr[0]) return false;
    if (prevAbs == currAbs) {
      if (prev[2] == curr[2]) return false;
    } else if (currAbs > prevAbs) {
      return false;
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(hasBalancedLedgerPostingOrder([]) == true);
  assert(hasBalancedLedgerPostingOrder(['AX:-4:C', 'BY:4:D']) == true);
  assert(hasBalancedLedgerPostingOrder(['AX:-4:C', 'BY:4:C']) == false);
  print('All tests passed!');
}