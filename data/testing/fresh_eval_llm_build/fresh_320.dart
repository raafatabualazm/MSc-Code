@pragma('vm:entry-point')
int countIngredientsSufficientForScale(List<String> ingredients, int scale) {
  if (scale <= 0) return 0;
  Map<String, int> freq = {};
  int count = 0;
  for (String item in ingredients) {
    int newCount = (freq[item] ?? 0) + 1;
    freq[item] = newCount;
    if (newCount == scale) count++;
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countIngredientsSufficientForScale(['flour', 'eggs', 'flour', 'sugar'], 1) == 3);
  assert(countIngredientsSufficientForScale(['flour', 'eggs', 'flour', 'sugar'], 2) == 1);
  assert(countIngredientsSufficientForScale([], 2) == 0);
  print('All tests passed!');
}