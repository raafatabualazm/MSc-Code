@pragma('vm:entry-point')
List<int> condenseScaledIngredientBatches(List<int> ingredientUnits, int scaleFactor) {
  List<int> adjusted = [];
  for (int i = 0; i < ingredientUnits.length; i++) {
    int scaled = ingredientUnits[i] * scaleFactor;
    if (scaled < 0) scaled = -scaled;
    if (scaled % 5 == 0) {
      scaled = scaled ~/ 5;
    } else if (scaled % 2 == 0) {
      scaled += scaleFactor;
    } else {
      scaled -= 1;
    }
    adjusted.add(scaled);
  }
  List<int> result = [];
  Set<int> seen = {};
  for (int value in adjusted) {
    if (value == 0 && !seen.contains(-1)) {
      result.add(0);
      seen.add(-1);
    } else if (value != 0 && !seen.contains(value)) {
      result.add(value);
      seen.add(value);
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(condenseScaledIngredientBatches([5, 5, 2], 1).toString() == '[1, 3]');
  assert(condenseScaledIngredientBatches([], 4).length == 0);
  assert(condenseScaledIngredientBatches([1, 2, 3, 4], 0).toString() == '[0]');
  print('All tests passed!');
}