@pragma('vm:never-inline')
@pragma('vm:entry-point')
String candidate(List<String> violations) {
  if (violations.isEmpty) return '';
  int severityRank(String msg) {
    if (msg.startsWith('[CRITICAL]')) return 0;
    if (msg.startsWith('[WARNING]')) return 1;
    if (msg.startsWith('[INFO]')) return 2;
    return 3;
  }
  final sorted = List<String>.from(violations);
  sorted.sort((a, b) {
    final rankA = severityRank(a);
    final rankB = severityRank(b);
    if (rankA != rankB) return rankA.compareTo(rankB);
    return a.compareTo(b);
  });
  return sorted.join('|');
}

void main() {
  final implementation = candidate;

  expect(implementation([]), '');
  expect(implementation(['[INFO] too short']), '[INFO] too short');
  expect(implementation(['[CRITICAL] no uppercase', '[CRITICAL] no symbol']), '[CRITICAL] no symbol|[CRITICAL] no uppercase');
  expect(implementation(['[INFO] too short', '[CRITICAL] no uppercase', '[WARNING] no digit', '[CRITICAL] no symbol']), '[CRITICAL] no symbol|[CRITICAL] no uppercase|[WARNING] no digit|[INFO] too short');
  expect(implementation(['[WARNING] no digit', '[WARNING] all lowercase', '[INFO] short']), '[WARNING] all lowercase|[WARNING] no digit|[INFO] short');
  expect(implementation(['[INFO] b', '[INFO] a', '[CRITICAL] z', '[WARNING] m']), '[CRITICAL] z|[WARNING] m|[INFO] a|[INFO] b');
  expect(implementation(['[UNKNOWN] custom rule', '[CRITICAL] no uppercase']), '[CRITICAL] no uppercase|[UNKNOWN] custom rule');
  expect(implementation(['[WARNING] only one']), '[WARNING] only one');
  expect(implementation(['[INFO] c', '[CRITICAL] a', '[INFO] a', '[WARNING] b']), '[CRITICAL] a|[WARNING] b|[INFO] a|[INFO] c');
}

void expect(dynamic a, dynamic b) {
  if (a == b) return;

  if (a is List && b is List) {
    expectList(a, b);
  } else if (a is Map && b is Map) {
    expectMap(a, b);
  } else {
    throw '$a != $b';
  }
}

void expectList(List a, List b) {
  if (a.length != b.length) throw 'list lengths are not equal';

  for (var i = 0; i < a.length; i++) {
    expect(a[i], b[i]);
  }
}

void expectMap(Map a, Map b) {
  if (a.length != b.length) throw 'map lengths are not equal';

  for (var key in a.keys) {
    expect(a[key], b[key]);
  }
}
