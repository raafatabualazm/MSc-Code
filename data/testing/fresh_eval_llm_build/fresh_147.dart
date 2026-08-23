@pragma('vm:entry-point')
int largestDenseHashBand(List<int> buckets) {
  if (buckets.isEmpty) {
    return 0;
  }
  int low = 1;
  int high = buckets.length;
  int answer = 0;
  while (low <= high) {
    int mid = (low + high) >> 1;
    bool found = false;
    for (int start = 0; start + mid <= buckets.length; start++) {
      bool usedRepair = false;
      bool valid = true;
      for (int i = start; i < start + mid; i++) {
        if (buckets[i] >= mid) {
          continue;
        }
        bool repaired = false;
        if (!usedRepair) {
          if (i > start && buckets[i - 1] + buckets[i] >= mid * 2) {
            repaired = true;
          } else if (i + 1 < start + mid && buckets[i + 1] + buckets[i] >= mid * 2) {
            repaired = true;
          }
        }
        if (!repaired) {
          valid = false;
          break;
        }
        usedRepair = true;
      }
      if (valid) {
        found = true;
        break;
      }
    }
    if (found) {
      answer = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return answer;
}

@pragma('vm:entry-point')
void main() {
  assert(largestDenseHashBand([]) == 0);
  assert(largestDenseHashBand([3, 1]) == 2);
  assert(largestDenseHashBand([4, 1, 4, 4]) == 2);
  print('All tests passed!');
}