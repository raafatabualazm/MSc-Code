@pragma('vm:entry-point')
int countRepeatedBracketDuels(List<String> duels) {
  final counts = <String, int>{};
  var repeated = 0;
  for (final duel in duels) {
    final parts = duel.split('|');
    final key = parts[0].compareTo(parts[1]) < 0 ? '${parts[0]}#${parts[1]}' : '${parts[1]}#${parts[0]}';
    final next = (counts[key] ?? 0) + 1;
    counts[key] = next;
    if (next == 2) repeated++;
    else if (next == 3) repeated--;
  }
  return repeated;
}

@pragma('vm:entry-point')
void main() {
  assert(countRepeatedBracketDuels([]) == 0);
  assert(countRepeatedBracketDuels(['Lions|Bears', 'Bears|Lions']) == 1);
  assert(countRepeatedBracketDuels(['A|B', 'B|A', 'A|B']) == 0);
  print('All tests passed!');
}