@pragma('vm:entry-point')
int countTwinCoreDnaTokens(String stream) {
  int count = 0;
  int start = 0;
  for (int i = 0; i <= stream.length; i++) {
    if (i == stream.length || stream[i] == '|') {
      String token = stream.substring(start, i);
      if (token.length == 4 && token[0] == 'A' && token[3] == 'T' && token[1] == token[2]) count++;
      start = i + 1;
    }
  }
  return count;
}

@pragma('vm:entry-point')
void main() {
  assert(countTwinCoreDnaTokens('ACCT') == 1);
  assert(countTwinCoreDnaTokens('AGCT|ACCA') == 0);
  assert(countTwinCoreDnaTokens('|AGGT||ATTT|') == 2);
  print('All tests passed!');
}