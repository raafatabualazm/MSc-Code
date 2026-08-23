@pragma('vm:entry-point')
List<String> samePrefixGenomeIntervals(List<String> samples) {
  List<String> out = [];
  for (int i = 1; i < samples.length; i++) {
    if (samples[i].isNotEmpty &&
        samples[i - 1].isNotEmpty &&
        samples[i][0] == samples[i - 1][0]) {
      out.add('${i - 1}-${i}:${(samples[i].length - samples[i - 1].length).abs()}');
    }
  }
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(samePrefixGenomeIntervals(['A', 'AA']).toString() == '[0-1:1]');
  assert(samePrefixGenomeIntervals(['AG', 'CT']).isEmpty);
  assert(samePrefixGenomeIntervals(['TT', 'T', 'TA']).toString() == '[0-1:1, 1-2:1]');
  print('All tests passed!');
}