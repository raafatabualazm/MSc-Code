@pragma('vm:entry-point')
int decipherMoistureRunlengthSum(String data) {
  int total = 0;
  for (int i = 0; i < data.length; i += 2) {
    int code = data.codeUnitAt(i);
    int value;
    if (code == 69) {
      value = 1;
    } else if (code == 78) {
      value = 2;
    } else if (code == 88) {
      value = 3;
    } else {
      throw ArgumentError('Invalid shifted category');
    }
    int count = int.parse(data[i + 1]);
    total += value * count;
  }
  return total;
}

@pragma('vm:entry-point')
void main() {
  assert(candidate("E1") == 1);
  assert(candidate("X4") == 12);
  assert(candidate("") == 0);
  print('All tests passed!');
}