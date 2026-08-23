@pragma('vm:entry-point')
bool isLogLineContentSafe(String logLine, String blocklist) {
  int spaceCount = 0;
  int messageStart = 0;
  for (int i = 0; i < logLine.length; i++) {
    if (logLine[i] == ' ') {
      spaceCount++;
      if (spaceCount == 3) {
        messageStart = i + 1;
        break;
      }
    }
  }
  if (spaceCount < 3) return false;
  String message = logLine.substring(messageStart);
  List<String> badWords = [];
  String current = '';
  for (int i = 0; i < blocklist.length; i++) {
    if (blocklist[i] == ',') {
      if (current.isNotEmpty) {
        badWords.add(current);
        current = '';
      }
    } else {
      current += blocklist[i];
    }
  }
  if (current.isNotEmpty) badWords.add(current);
  List<String> words = [];
  current = '';
  for (int i = 0; i < message.length; i++) {
    if (message[i] == ' ') {
      if (current.isNotEmpty) {
        words.add(current);
        current = '';
      }
    } else {
      current += message[i];
    }
  }
  if (current.isNotEmpty) words.add(current);
  for (String word in words) {
    for (String bad in badWords) {
      if (word.length == bad.length) {
        bool match = true;
        for (int i = 0; i < word.length; i++) {
          if (word[i] != bad[i]) {
            match = false;
            break;
          }
        }
        if (match) return false;
      }
    }
  }
  return true;
}

@pragma('vm:entry-point')
void main() {
  assert(isLogLineContentSafe('2023-01-01 DEBUG main log message', 'error,warning') == true);
  assert(isLogLineContentSafe('2023-01-01 DEBUG main error occurred', 'error,warning') == false);
  assert(isLogLineContentSafe('', 'x') == false);
  print('All tests passed!');
}