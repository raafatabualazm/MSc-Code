@pragma('vm:entry-point')
int recountResidue(int tallyTape) {
  int sum = 0;
  int weight = 3;
  while (tallyTape > 0) {
    sum += (tallyTape % 10) * weight;
    weight = weight == 3 ? 1 : 3;
    tallyTape ~/= 10;
  }
  return sum % 11;
}

@pragma('vm:entry-point')
void main() {
  assert(recountResidue(0) == 0);
  assert(recountResidue(12345) == 0);
  assert(recountResidue(909) == 10);
  print('All tests passed!');
}