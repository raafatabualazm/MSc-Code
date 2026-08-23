@pragma('vm:entry-point')
String bestHalfWindowSum(List<int> sizes) {
  int n = sizes.length;
  if (n < 2) return "0:0";
  int k = n ~/ 2, sum = 0, maxSum = 0, maxStart = 0;
  for (int i = 0; i < n; i++) {
    sum += sizes[i];
    if (i >= k) sum -= sizes[i - k];
    if (i == k - 1) {
      maxSum = sum; maxStart = 0;
    } else if (i >= k && sum > maxSum) {
      maxSum = sum; maxStart = i - k + 1;
    }
  }
  return "${maxSum}:${maxStart}";
}

@pragma('vm:entry-point')
void main() {
  assert(bestHalfWindowSum([1,2,3,4]) == "7:2");
  assert(bestHalfWindowSum([]) == "0:0");
  assert(bestHalfWindowSum([5,1,3,10,2]) == "13:2");
  print('All tests passed!');
}