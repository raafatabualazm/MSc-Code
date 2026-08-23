@pragma('vm:entry-point')
int countFragmentedRows(List<String> seating) {
  int count = 0;
  for (int i = 0; i < seating.length; i++) {
    String row = seating[i];
    int state = 0;
    bool frag = false;
    for (int j = 0; j < row.length; j++) {
      if (row[j] == '#') {
        if (state == 2) {
          frag = true;
          break;
        }
        state = 1;
      } else if (row[j] == '.') {
        if (state == 1) {
          state = 2;
        }
      }
    }
    if (frag) count++;
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countFragmentedRows([]) == 0);
  assert(countFragmentedRows(['.#.']) == 0);
  assert(countFragmentedRows(['#.#']) == 1);
  print('All tests passed!');
}