@pragma('vm:entry-point')
bool formsLayeredTideCrest(List<int> readings) {
  bool check(int left, int right) {
    if (right - left < 2) {
      return true;
    }
    if (readings[left] + 2 != readings[left + 1] ||
        readings[right] + 2 != readings[right - 1]) {
      return false;
    }
    return check(left + 1, right - 1);
  }

  return check(0, readings.length - 1);
}

@pragma('vm:entry-point')
void main() {
  assert(formsLayeredTideCrest([]) == true);
  assert(formsLayeredTideCrest([1, 3, 5, 3, 1]) == true);
  assert(formsLayeredTideCrest([1, 3, 4, 3, 1]) == false);
  print('All tests passed!');
}