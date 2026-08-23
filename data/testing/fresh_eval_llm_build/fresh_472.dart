@pragma('vm:entry-point')
List<num> summarizeInventoryCrates(List<String> logs) {
  Map<String, int> counts = {};
  int disruption = 0;
  for (String log in logs) {
    bool cracked = log.endsWith('*');
    String item = cracked ? log.substring(0, log.length - 1) : log;
    int next = (counts[item] ?? 0) + (cracked ? -1 : 1);
    if (next <= 0) {
      counts.remove(item);
      disruption += cracked ? 2 : 1;
    } else {
      counts[item] = next;
      if (next >= 4) {
        disruption += item.length.isEven ? 1 : 2;
      } else if (cracked) {
        disruption += 1;
      }
    }
  }
  int bulky = 0;
  double singles = 0.0;
  for (int amount in counts.values) {
    if (amount == 1) {
      singles += 0.5;
    } else if (amount >= 3) {
      bulky++;
    }
  }
  return [counts.length, bulky, singles + disruption];
}

@pragma('vm:entry-point')
void main() {
  assert(summarizeInventoryCrates([]).toString() == '[0, 0, 0.0]');
  assert(summarizeInventoryCrates(['orb', 'orb', 'orb', 'orb']).toString() == '[1, 1, 2.0]');
  assert(summarizeInventoryCrates(['ring', 'ring', 'ring*']).toString() == '[1, 0, 1.5]');
  print('All tests passed!');
}