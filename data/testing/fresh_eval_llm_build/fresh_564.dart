@pragma('vm:entry-point')
bool validateLedgerTape(String ledger, int openingBalance) {
  int balance = openingBalance;
  if (ledger.isEmpty) return balance == 0;
  for (String entry in ledger.split(',')) {
    if (entry.isEmpty) return false;
    int i = 1, amount = 0, digits = 0;
    String type = entry[0];
    if (type != 'D' && type != 'W') return false;
    while (i < entry.length) {
      int c = entry.codeUnitAt(i);
      if (c < 48 || c > 57) break;
      amount = amount * 10 + c - 48;
      digits++;
      i++;
    }
    if (digits == 0 || (digits > 1 && entry[1] == '0')) return false;
    if (i >= entry.length || entry[i++] != '#') return false;
    int vowels = 0, letters = 0;
    while (i < entry.length) {
      String ch = entry[i++];
      if (ch.codeUnitAt(0) < 97 || ch.codeUnitAt(0) > 122) return false;
      if ('aeiou'.contains(ch)) vowels++;
      letters++;
      if (type == 'W' && letters > amount) return false;
    }
    if (letters == 0) return false;
    if ((amount + vowels).isOdd) continue;
    balance += type == 'D' ? amount : -amount;
    if (balance < 0) return false;
  }
  return balance == 0;
}

@pragma('vm:entry-point')
void main() {
  assert(validateLedgerTape('', 0) == true);
  assert(validateLedgerTape('D2#aa,W2#bb', 0) == true);
  assert(validateLedgerTape('W2#aa', 0) == false);
  print('All tests passed!');
}