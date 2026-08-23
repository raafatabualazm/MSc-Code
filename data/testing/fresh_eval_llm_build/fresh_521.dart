@pragma('vm:entry-point')
String? replaySpellCheckerEdits(String seed, List<String> edits) {
  String text = seed;
  List<String> undo = [];
  for (String edit in edits) {
    if (edit == 'UNDO') {
      if (undo.isNotEmpty) {
        text = undo.removeLast();
      }
      continue;
    }
    if (edit.startsWith('+')) {
      undo.add(text);
      text += edit.substring(1);
    } else if (edit.startsWith('-')) {
      int count = int.tryParse(edit.substring(1)) ?? -1;
      if (count < 0 || count > text.length) return null;
      undo.add(text);
      text = text.substring(0, text.length - count);
    } else if (edit.startsWith('*')) {
      int cut = edit.indexOf(':');
      if (cut < 2) return null;
      String from = edit.substring(1, cut);
      String to = edit.substring(cut + 1);
      bool changed = false;
      for (int i = text.length - from.length; i >= 0; i--) {
        int j = 0;
        while (j < from.length && text[i + j] == from[j]) {
          j++;
        }
        if (j == from.length) {
          undo.add(text);
          text = text.substring(0, i) + to + text.substring(i + from.length);
          changed = true;
          break;
        }
      }
      if (!changed) return null;
    } else {
      return null;
    }
  }
  return text;
}

@pragma('vm:entry-point')
void main() {
  assert(replaySpellCheckerEdits('', ['+teh', '*teh:the']) == 'the');
  assert(replaySpellCheckerEdits('speling', ['*speling:spelling', '-3']) == 'spell');
  assert(replaySpellCheckerEdits('cat', ['UNDO']) == 'cat');
  print('All tests passed!');
}