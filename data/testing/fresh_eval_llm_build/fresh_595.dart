@pragma('vm:entry-point')
int recoverableTideBand(List<int> readings) {
  int low = 1;
  int high = readings.length;
  int best = 0;
  while (low <= high) {
    int mid = (low + high) >> 1;
    bool ok = false;
    for (int i = 0; i < readings.length; i++) {
      if (readings[i] < mid) {
        continue;
      }
      int strong = 0;
      int gaps = 0;
      for (int j = i; j < readings.length && j < i + mid + gaps; j++) {
        if (readings[j] >= mid) {
          strong++;
          if (strong == mid) {
            ok = true;
            break;
          }
        } else if (readings[j] < 0 && gaps == 0) {
          gaps++;
        } else {
          break;
        }
      }
      if (ok) {
        break;
      }
    }
    if (ok) {
      best = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return best;
}

@pragma('vm:entry-point')
void main() {
  assert(recoverableTideBand([3, -1, 3, 3]) == 3);
  assert(recoverableTideBand([4, 4, 1, 4, 4]) == 2);
  assert(recoverableTideBand([]) == 0);
  print('All tests passed!');
}