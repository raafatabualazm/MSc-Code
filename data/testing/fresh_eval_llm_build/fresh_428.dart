@pragma('vm:entry-point')
int hashBucketRunChecksum(String encoded, int bucketCount) {
  if (bucketCount <= 0) return -1;
  List<int> buckets = List.filled(bucketCount, 0);
  int score = 0;
  int i = 0;
  while (i < encoded.length) {
    int first = encoded.codeUnitAt(i);
    if (first >= 48 && first <= 57) {
      score -= 2;
      i++;
      continue;
    }
    int repeat = 0;
    int j = i + 1;
    while (j < encoded.length) {
      int code = encoded.codeUnitAt(j);
      if (code < 48 || code > 57) break;
      repeat = repeat * 10 + code - 48;
      j++;
    }
    if (repeat == 0) repeat = 1;
    for (int k = 0; k < repeat; k++) {
      int bucket = (first + k) % bucketCount;
      if (encoded[i] == '#') {
        buckets[bucket] = 0;
        score += bucket;
        continue;
      } else if (encoded[i].toUpperCase() == encoded[i] && encoded[i].toLowerCase() != encoded[i]) {
        buckets[bucket] += 2;
      } else {
        buckets[bucket] += 1;
      }
      if (buckets[bucket] == 2) {
        score += 3;
      } else if (buckets[bucket] > 3) {
        score -= bucket + 1;
      } else {
        score += bucket;
      }
    }
    i = j;
  }
  for (final v in buckets) {
    score += v;
  }
  return score;
}

@pragma('vm:entry-point')
void main() {
  assert(hashBucketRunChecksum('', 3) == 0);
  assert(hashBucketRunChecksum('A', 2) == 5);
  assert(hashBucketRunChecksum('a10', 2) == 12);
  print('All tests passed!');
}