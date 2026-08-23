@pragma('vm:entry-point')
String formatPrecinctsAboveThreshold(List<int> votes, int threshold) {
  String precincts = "";
  int total = 0;
  for (int i = 0; i < votes.length; i++) {
    if (votes[i] >= threshold) {
      if (precincts.isNotEmpty) precincts += ", ";
      precincts += "$i";
      total += votes[i];
    }
  }
  return "Precincts above threshold: $precincts; Total votes: $total";
}

@pragma('vm:entry-point')
void main() {
  assert(formatPrecinctsAboveThreshold([], 5) == "Precincts above threshold: ; Total votes: 0");
  assert(formatPrecinctsAboveThreshold([7, 3], 4) == "Precincts above threshold: 0; Total votes: 7");
  assert(formatPrecinctsAboveThreshold([2, 2, 2], 3) == "Precincts above threshold: ; Total votes: 0");
  print('All tests passed!');
}