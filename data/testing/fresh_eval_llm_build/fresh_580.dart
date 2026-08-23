@pragma('vm:entry-point')
int countBucketCascadePenalty(List<int> buckets) {
  int solve(int start, int end) {
    if (start >= end) {
      if (start == end && buckets[start] > 3) {
        return 1;
      }
      return 0;
    }
    int peak = start;
    int spread = 0;
    bool hasGap = false;
    for (int i = start; i <= end; i++) {
      if (buckets[i] > buckets[peak]) {
        peak = i;
      }
      for (int j = i + 1; j <= end; j++) {
        int d = (buckets[i] - buckets[j]).abs();
        if (d > spread) {
          spread = d;
        }
        if (d > 2) {
          hasGap = true;
        }
      }
    }
    if (spread <= 1) {
      return end - start + 1;
    }
    int score = hasGap ? spread : 0;
    if (peak > start) {
      score += solve(start, peak - 1);
    }
    if (peak < end) {
      score += solve(peak + 1, end);
    }
    if (buckets[peak] % 2 == 0) {
      score += 1;
    } else if (buckets[peak] == 1) {
      score -= 1;
    }
    return score;
  }

  if (buckets.isEmpty) {
    return 0;
  }
  return solve(0, buckets.length - 1);
}

@pragma('vm:entry-point')
void main() {
  assert(countBucketCascadePenalty([]) == 0);
  assert(countBucketCascadePenalty([2, 2]) == 2);
  assert(countBucketCascadePenalty([0, 4, 0, 4]) == 10);
  print('All tests passed!');
}