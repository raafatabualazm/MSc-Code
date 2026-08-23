@pragma('vm:entry-point')
bool hasSingleAmberHold(List<int> phases) {
  if (phases.length < 2) return false;
  int base = phases[0];
  int left = 0;
  int right = phases.length - 1;
  while (left < right) {
    int mid = (left + right) >> 1;
    int expected = base + mid * 5;
    if (phases[mid] == expected) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  int pivot = left;
  if (pivot == 0 || phases[pivot] != base + (pivot - 1) * 5) return false;
  if (phases[pivot - 1] != phases[pivot]) return false;
  for (int i = pivot + 1; i < phases.length; i++) {
    if (phases[i] != base + (i - 1) * 5) return false;
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(hasSingleAmberHold([10, 15, 15, 20, 25]) == true);
  assert(hasSingleAmberHold([10, 15, 20, 25]) == false);
  assert(hasSingleAmberHold([30, 30, 35]) == true);
  print('All tests passed!');
}