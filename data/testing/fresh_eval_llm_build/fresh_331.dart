@pragma('vm:entry-point')
List<int> processTransactionRounds(List<String> transactions) {
  int roundingIncrement = 1;
  List<int> results = [];
  for (var txn in transactions) {
    int subTotal = 0;
    var parts = txn.split(' ');
    for (var part in parts) {
      if (part.isEmpty) continue;
      if (part.startsWith('ROUND:')) {
        var val = int.tryParse(part.substring(6));
        if (val != null && val > 0) roundingIncrement = val;
      } else {
        var amt = int.tryParse(part);
        if (amt != null) subTotal += amt;
      }
    }
    int rounded = ((subTotal + roundingIncrement ~/ 2) ~/ roundingIncrement) * roundingIncrement;
    results.add(rounded);
  }
  return results;
}

@pragma('vm:entry-point')
void main() {
  assert(candidate([]).isEmpty);
  assert(candidate(["100 200"]).first == 300);
  assert(candidate(["ROUND:5 12"]).first == 10);
  print('All tests passed!');
}