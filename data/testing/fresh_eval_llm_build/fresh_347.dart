@pragma('vm:entry-point')
bool isLibraryShelfTrailCode(String code) {
  int state = 0;
  for (int i = 0; i < code.length; i++) {
    int c = code.codeUnitAt(i);
    if ((state == 0 && c >= 65 && c <= 90) ||
        ((state == 1 || state == 2) && c >= 48 && c <= 57) ||
        (state == 3 && c == 58)) {
      state = state == 0 ? 1 : state == 1 ? 2 : state == 2 ? 3 : 0;
    } else {
      return false;
    }
  }
  return state == 3;
}

@pragma('vm:entry-point')
void main() {
  assert(isLibraryShelfTrailCode('A12') == true);
  assert(isLibraryShelfTrailCode('A12:B34') == true);
  assert(isLibraryShelfTrailCode('A1') == false);
  print('All tests passed!');
}