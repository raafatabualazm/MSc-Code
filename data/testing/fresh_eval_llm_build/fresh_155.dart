@pragma('vm:entry-point')
List<int> inventoryFusionProgress(List<int> itemLevels) {
  List<int> dp = List<int>.filled(itemLevels.length, 0);
  for (int i = 0; i < itemLevels.length; i++) {
    int base = itemLevels[i];
    if (base < 0) {
      base = 0;
    }
    dp[i] = base;
    if (i > 0) {
      if (itemLevels[i] == itemLevels[i - 1]) {
        dp[i] = dp[i - 1] + itemLevels[i] + 1;
      } else if (itemLevels[i] > itemLevels[i - 1]) {
        int gain = dp[i - 1] + itemLevels[i];
        if (gain > dp[i]) dp[i] = gain;
      } else if (itemLevels[i - 1] - itemLevels[i] == 1) {
        int gain = dp[i - 1] - 1;
        if (gain > dp[i]) dp[i] = gain;
      }
      if (dp[i] < dp[i - 1] && itemLevels[i] >= 0) {
        dp[i] = dp[i - 1];
      }
    }
  }
  return dp;
}

@pragma('vm:entry-point')
void main() {
  assert(inventoryFusionProgress([2, 2]).toString() == '[2, 5]');
  assert(inventoryFusionProgress([-1, 2]).toString() == '[0, 2]');
  assert(inventoryFusionProgress([3, 1, 2]).toString() == '[3, 3, 5]');
  print('All tests passed!');
}