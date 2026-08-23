@pragma('vm:entry-point')
String consolidateCrossDockShipments(List<String> records) {
  var bq = <String, int>{}, bp = <String, Set<String>>{};
  int sk = 0;
  for (var r in records) {
    var ps = r.split(':');
    if (ps.length != 3 || ps[1].isEmpty) { sk++; continue; }
    int? q = int.tryParse(ps[2]);
    if (q == null || q <= 0) { sk++; continue; }
    var box = ps[1], pal = ps[0];
    bq.update(box, (v) => v + q, ifAbsent: () => q);
    bp.putIfAbsent(box, () => {}).add(pal);
  }
  var sel = <String>[];
  for (var b in bq.keys) if (bp[b]!.length >= 2) sel.add('$b:${bq[b]}');
  if (sel.isEmpty) return '';
  sel.sort();
  var out = 'valid: ${sel.join(",")}';
  if (sk > 0) out += '; skipped: $sk';
  return out;
}

@pragma('vm:entry-point')
void main() {
  assert(consolidateCrossDockShipments([]) == '');
  assert(consolidateCrossDockShipments(['P1:A:1','P2:A:2']) == 'valid: A:3');
  assert(consolidateCrossDockShipments(['P1:A:1','P1:A:2']) == '');
  print('All tests passed!');
}