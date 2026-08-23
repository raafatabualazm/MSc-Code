@pragma('vm:entry-point')
List<String> prioritizeSpellEdits(List<String> edits) {
  List<String> result = List<String>.from(edits);
  result.sort((a, b) {
    List<String> pa = a.split('->');
    List<String> pb = b.split('->');
    int gapA = (pa[0].length - pa[1].length).abs();
    int gapB = (pb[0].length - pb[1].length).abs();
    if (gapA != gapB) return gapA - gapB;
    bool sameA = pa[0].isNotEmpty && pa[1].isNotEmpty && pa[0][0] == pa[1][0];
    bool sameB = pb[0].isNotEmpty && pb[1].isNotEmpty && pb[0][0] == pb[1][0];
    if (sameA != sameB) return sameA ? -1 : 1;
    if (pa[0].length != pb[0].length) return pa[0].length - pb[0].length;
    return a.compareTo(b);
  });
  return result;
}

@pragma('vm:entry-point')
void main() {
  assert(prioritizeSpellEdits([]).toString() == '[]');
  assert(prioritizeSpellEdits(['teh->the', 'bok->book']).toString() == '[teh->the, bok->book]');
  assert(prioritizeSpellEdits(['cat->dog', 'tap->top']).toString() == '[tap->top, cat->dog]');
  print('All tests passed!');
}