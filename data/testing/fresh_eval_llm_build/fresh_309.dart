@pragma('vm:entry-point')
int countRecipeScaleSpikes(String plan) {
  if (plan.isEmpty) return 0;
  int spikes = 0;
  for (final item in plan.split(';')) {
    final parts = item.split(':');
    final amounts = parts[1].split('>');
    final base = int.parse(amounts[0]);
    final scaled = int.parse(amounts[1]);
    if (scaled > base * 2) spikes++;
  }
  return spikes;
}

@pragma('vm:entry-point')
void main() {
  assert(countRecipeScaleSpikes('sugar:1>3') == 1);
  assert(countRecipeScaleSpikes('flour:2>4;milk:3>6') == 0);
  assert(countRecipeScaleSpikes('salt:0>1;yeast:4>9') == 2);
  print('All tests passed!');
}