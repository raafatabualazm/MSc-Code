@pragma('vm:entry-point')
String longestLeadingVoteSegment(String votes) {
  if (votes.isEmpty) return "";
  int n = votes.length;
  int maxLen = 0;
  int startIdx = 0;
  for (int i = 0; i < n; i++) {
    int sum = 0;
    int j = i;
    while (j < n) {
      sum += (votes[j] == 'A' ? 1 : -1);
      if (sum < 0) break;
      j++;
    }
    int len = j - i;
    if (len > maxLen) {
      maxLen = len;
      startIdx = i;
    }
  }
  return votes.substring(startIdx, startIdx + maxLen);
}

@pragma('vm:entry-point')
void main() {
  assert(longestLeadingVoteSegment("") == "");
  assert(longestLeadingVoteSegment("A") == "A");
  assert(longestLeadingVoteSegment("AB") == "AB");
  print('All tests passed!');
}