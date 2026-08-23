@pragma('vm:entry-point')
String reconcileRoundedCentBuckets(List<int> cents) {
  if (cents.isEmpty) return 'empty';
  List<int> uniqueRounded = [];
  int up = 0;
  int down = 0;
  int flat = 0;
  int net = 0;
  for (int value in cents) {
    if (value == 0) {
      flat++;
      continue;
    }
    int sign = value < 0 ? -1 : 1;
    int absValue = value.abs();
    int remainder = absValue % 5;
    int rounded = remainder >= 3 ? absValue + (5 - remainder) : absValue - remainder;
    rounded *= sign;
    int diff = rounded - value;
    if (diff > 0) {
      up++;
    } else if (diff < 0) {
      down++;
    } else {
      flat++;
    }
    net += diff;
    bool seen = false;
    for (int existing in uniqueRounded) {
      if (existing == rounded) {
        seen = true;
        break;
      }
    }
    if (!seen) uniqueRounded.add(rounded);
  }
  uniqueRounded.sort();
  return '$up;$down;$flat;$net;${uniqueRounded.join(",")}';
}

@pragma('vm:entry-point')
void main() {
  assert(reconcileRoundedCentBuckets([]) == 'empty');
  assert(reconcileRoundedCentBuckets([3]) == '1;0;0;2;5');
  assert(reconcileRoundedCentBuckets([2, 3, 7, 8]) == '2;2;0;0;0,5,10');
  print('All tests passed!');
}