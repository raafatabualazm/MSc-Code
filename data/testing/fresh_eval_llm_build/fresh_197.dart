@pragma('vm:entry-point')
int calculateManifestHoldDays(List<int> manifests, int quarantineSpan) {
  if (quarantineSpan <= 0 || manifests.isEmpty) {
    return 0;
  }
  int total = 0;
  for (int i = 0; i < manifests.length; i++) {
    int day = manifests[i];
    if (day < 0) {
      continue;
    }
    for (int offset = 1; offset <= quarantineSpan; offset++) {
      int probe = day + offset;
      bool canceled = false;
      bool repeated = false;
      for (int j = 0; j < manifests.length; j++) {
        if (manifests[j] == -probe) {
          canceled = true;
          break;
        }
        if (manifests[j] == probe) {
          repeated = true;
        }
      }
      if (canceled) {
        continue;
      }
      if (repeated) {
        total += offset == quarantineSpan ? 2 : 1;
      } else if (probe % 7 == 0) {
        total++;
      }
    }
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(calculateManifestHoldDays([], 4) == 0);
  assert(calculateManifestHoldDays([5, 6], 1) == 3);
  assert(calculateManifestHoldDays([6, -7], 2) == 0);
  print('All tests passed!');
}