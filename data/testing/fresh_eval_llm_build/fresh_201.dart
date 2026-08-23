@pragma('vm:entry-point')
bool validateSoilMoistureRows(String scan, int width) {
  int cells = 0;
  for (int i = 0; i < scan.length; i++) {
    String c = scan[i];
    if (c == '/') {
      if (cells != width) return false;
      cells = 0;
    } else if (c == 'D' || c == 'M' || c == 'W') {
      cells++;
    } else {
      return false;
    }
  }
  return cells == width;
}

@pragma('vm:entry-point')
void main() {
  assert(validateSoilMoistureRows('DMW/WWD', 3) == true);
  assert(validateSoilMoistureRows('DM/WWD', 3) == false);
  assert(validateSoilMoistureRows('', 0) == true);
  print('All tests passed!');
}