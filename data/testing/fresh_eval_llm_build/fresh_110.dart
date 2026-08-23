@pragma('vm:entry-point')
bool areShelfCodesConsecutive(String code1, String code2) {
  int space1 = code1.indexOf(' ');
  if (space1 <= 0) return false;
  int space2 = code2.indexOf(' ');
  if (space2 <= 0) return false;

  for (int i = 0; i < space1; i++) {
    int cu = code1.codeUnitAt(i);
    if (cu < 65 || cu > 90) return false;
  }
  for (int i = 0; i < space2; i++) {
    int cu = code2.codeUnitAt(i);
    if (cu < 65 || cu > 90) return false;
  }

  String numStr1 = code1.substring(space1 + 1);
  String numStr2 = code2.substring(space2 + 1);
  if (numStr1.isEmpty || numStr2.isEmpty) return false;

  for (int i = 0; i < numStr1.length; i++) {
    int cu = numStr1.codeUnitAt(i);
    if (cu < 48 || cu > 57) return false;
  }
  for (int i = 0; i < numStr2.length; i++) {
    int cu = numStr2.codeUnitAt(i);
    if (cu < 48 || cu > 57) return false;
  }

  String cat1 = code1.substring(0, space1);
  String cat2 = code2.substring(0, space2);
  if (cat1 != cat2) return false;

  int num1 = int.parse(numStr1);
  int num2 = int.parse(numStr2);
  return (num1 - num2).abs() == 1;
}

@pragma('vm:entry-point')
void main() {
  assert(areShelfCodesConsecutive('FIC 100', 'FIC 101') == true);
  assert(areShelfCodesConsecutive('FIC 100', 'FIC 100') == false);
  assert(areShelfCodesConsecutive('FIC 100', 'SCI 101') == false);
  print('All tests passed!');
}