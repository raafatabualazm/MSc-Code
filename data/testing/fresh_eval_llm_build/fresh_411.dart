@pragma('vm:entry-point')
double accumulateEditCostForCandidate(String edits) {
  int length = 0;
  double cost = 0.0;
  for (int i = 0; i < edits.length; i++) {
    String c = edits[i];
    if (c == 'i') {
      length++;
      cost += 1.0;
    } else if (c == 'd') {
      if (length > 0) {
        length--;
        cost += 1.5;
      }
    } else if (c == 's') {
      if (length > 0) {
        cost += 2.0;
      }
    }
  }
  return cost;
}

@pragma('vm:entry-point')
void main() {
  assert(accumulateEditCostForCandidate("") == 0.0);
  assert(accumulateEditCostForCandidate("i") == 1.0);
  assert(accumulateEditCostForCandidate("id") == 2.5);
  print('All tests passed!');
}