@pragma('vm:entry-point')
List<int> restockInventory(List<int> inventory) {
  if (inventory.isEmpty) return [];
  int threshold = inventory[0];
  List<int> result = [];
  for (int i = 1; i < inventory.length; i++) {
    int item = inventory[i];
    result.add(item < threshold ? item * 2 : item ~/ 2);
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(restockInventory([3,2,3,5]).toString() == '[4, 1, 2]');
  assert(restockInventory([]).toString() == '[]');
  assert(restockInventory([10,9,8]).toString() == '[18, 16]');
  print('All tests passed!');
}