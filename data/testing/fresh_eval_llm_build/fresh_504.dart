@pragma('vm:entry-point')
List<String> reviewRoundedCentBuckets(List<int> cents) {
  if (cents.isEmpty) return ['empty'];
  List<String> out = [];
  for (int value in cents) {
    if (value == 0) {
      out.add('0:zero');
      continue;
    }
    int absValue = value.abs(), rem = absValue % 5;
    int rounded = rem < 3 ? absValue - rem : absValue + 5 - rem;
    if (value < 0) rounded = -rounded;
    int gap = (rounded - value).abs(), a = absValue, b = rounded.abs();
    while (b != 0) {
      int t = a % b;
      a = b;
      b = t;
    }
    bool prime = a > 1;
    for (int d = 2; d * d <= a; d++) {
      if (a % d == 0) {
        prime = false;
        break;
      }
    }
    if (gap == 0) {
      out.add('$value:locked:${prime ? 'p' : 'c'}');
    } else if (prime && rounded.isEven) {
      out.add('$value->$rounded:even');
    } else if (!prime && gap == 1) {
      out.add('$value->$rounded:step');
    } else {
      out.add('$value->$rounded:${a.toRadixString(16)}');
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(reviewRoundedCentBuckets([]).toString() == '[empty]');
  assert(reviewRoundedCentBuckets([12, 13]).toString() == '[12->10:even, 13->15:1]');
  assert(reviewRoundedCentBuckets([-14]).length == 1);
  print('All tests passed!');
}