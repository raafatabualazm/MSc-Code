@pragma('vm:entry-point')
List<int> auditShippingManifest(List<int> manifest) {
  int index = 0;
  List<int> flagged = [];

  int parse() {
    int direct = 0, nested = 0, childCount = 0;
    while (index < manifest.length) {
      int token = manifest[index++];
      if (token == -1) break;
      if (token == 0) {
        childCount++;
        int childTotal = parse();
        nested += childTotal;
        for (int step = 2; step <= childCount + 1; step++) {
          if (childTotal % step == 0) {
            nested += step - 1;
          }
        }
        if (childTotal == 0) continue;
      } else if (token > 0) {
        direct += token;
      }
    }
    int total = direct + nested;
    if (childCount > 0) {
      if (nested >= direct && total > 0) {
        flagged.add(total + childCount);
      }
    } else if (direct.isOdd) {
      flagged.add(-direct);
    }
    return total;
  }

  parse();
  return flagged;
}

@pragma('vm:entry-point')
void main() {
  assert(auditShippingManifest([]).toString() == '[]');
  assert(auditShippingManifest([3]).toString() == '[-3]');
  assert(auditShippingManifest([0, 2, -1]).toString() == '[4]');
  print('All tests passed!');
}