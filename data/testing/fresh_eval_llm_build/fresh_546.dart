@pragma('vm:entry-point')
double packetMaintenanceGapScore(List<int> packetSizes, int maintenanceDays) {
  if (maintenanceDays <= 0) {
    return -1.0;
  }
  if (packetSizes.isEmpty) {
    return 0.0;
  }
  int score = 0;
  for (int i = 0; i < packetSizes.length; i++) {
    if (packetSizes[i] < 0) {
      score -= 2;
      continue;
    }
    int streak = 0;
    for (int gap = 1; gap <= maintenanceDays && i + gap < packetSizes.length; gap++) {
      int future = packetSizes[i + gap];
      if (future < 0) {
        streak -= gap;
        break;
      }
      int diff = (future - packetSizes[i]).abs();
      if (diff == 0) {
        streak += 2;
        continue;
      }
      if (diff <= gap * 8) {
        streak += 1;
      } else if (diff % maintenanceDays == 0) {
        streak -= 2;
      } else {
        streak -= 1;
      }
    }
    if ((i + 1) % maintenanceDays == 0) {
      score += streak + 1;
    } else if (streak > 0) {
      score += streak;
    } else {
      score -= 1;
    }
  }
  return score / 2.0;
}

@pragma('vm:entry-point')
void main() {
  assert(packetMaintenanceGapScore([], 3) == 0.0);
  assert(packetMaintenanceGapScore([4], 1) == 0.5);
  assert(packetMaintenanceGapScore([2, 10], 2) == 1.0);
  print('All tests passed!');
}