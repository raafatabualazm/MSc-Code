@pragma('vm:entry-point')
String locateMedianTrafficTag(List<String> lines) {
  if (lines.isEmpty) return 'no-logs';
  List<String> tags = [];
  List<int> counts = [];
  int total = 0;
  for (String line in lines) {
    int split = line.indexOf('#');
    if (split <= 0 || split != line.lastIndexOf('#') || split == line.length - 1) {
      return 'invalid';
    }
    String tag = line.substring(0, split);
    int? count = int.tryParse(line.substring(split + 1));
    if (count == null || count < 0) return 'invalid';
    tags.add(tag);
    counts.add(count);
    total += count;
  }
  if (total == 0) return 'empty-traffic';
  int target = (total + 1) ~/ 2;
  int low = 0;
  int high = counts.length - 1;
  while (low <= high) {
    int mid = (low + high) ~/ 2;
    int prefix = 0;
    for (int i = 0; i <= mid; i++) {
      prefix += counts[i];
      if (prefix >= target) break;
    }
    int before = prefix - counts[mid];
    if (before >= target) {
      high = mid - 1;
    } else if (prefix < target) {
      low = mid + 1;
    } else {
      if (total % 2 == 0 && prefix == total ~/ 2 && mid + 1 < tags.length) {
        return tags[mid] + '|' + tags[mid + 1];
      }
      return tags[mid];
    }
  }
  return 'invalid';
}

@pragma('vm:entry-point')
void main() {
  assert(locateMedianTrafficTag([]) == 'no-logs');
  assert(locateMedianTrafficTag(['api#2','db#3','cdn#5']) == 'db|cdn');
  assert(locateMedianTrafficTag(['edge#1','core#4']) == 'core');
  print('All tests passed!');
}