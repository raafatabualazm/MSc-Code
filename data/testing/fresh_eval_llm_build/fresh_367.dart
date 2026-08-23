@pragma('vm:entry-point')
bool hasLayeredTransferMinutes(List<int> minutes) {
  bool solve(List<int> part) {
    if (part.length < 2) {
      return true;
    }
    for (int pivot = 0; pivot < part.length; pivot++) {
      int hub = part[pivot];
      List<int> earlier = [];
      List<int> later = [];
      bool ok = true;
      for (int i = 0; i < part.length; i++) {
        if (i == pivot) continue;
        int gap = (part[i] - hub).abs();
        if (gap == 0 || gap > 30) {
          ok = false;
          break;
        }
        if (part[i] < hub) {
          for (int v in earlier) {
            if (v == part[i]) return false;
          }
          earlier.add(part[i]);
        } else {
          for (int v in later) {
            if (v == part[i]) return false;
          }
          later.add(part[i]);
        }
      }
      if (ok && solve(earlier) && solve(later)) {
        return true;
      }
    }
    return false;
  }

  return solve(minutes);
}

@pragma('vm:entry-point')
void main() {
  assert(hasLayeredTransferMinutes([]) == true);
  assert(hasLayeredTransferMinutes([0, 25, 55]) == true);
  assert(hasLayeredTransferMinutes([0, 40, 70]) == false);
  print('All tests passed!');
}