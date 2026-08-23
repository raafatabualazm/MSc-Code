@pragma('vm:entry-point')
List<int> interiorGapWidths(List<int> occupiedSeats) {
  if (occupiedSeats.length < 2) return [];
  List<int> sorted = List<int>.from(occupiedSeats)..sort();
  List<int> result = [];
  for (int i = 0; i < sorted.length - 1; i++) {
    int gap = sorted[i + 1] - sorted[i] - 1;
    if (gap > 0) {
      if (gap == 1) {
        result.add(2);
      } else {
        result.add(gap);
      }
    }
  }
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(interiorGapWidths([]).toString() == [].toString());
  assert(interiorGapWidths([2, 5, 9, 11]).toString() == [2, 3, 2].toString());
  assert(interiorGapWidths([1, 3]).toString() == [2].toString());
  print('All tests passed!');
}