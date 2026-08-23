@pragma('vm:entry-point')
String executeThermostatCommands(List<String> commands, double initialSetpoint) {
  String mode = 'OFF';
  double setpoint = initialSetpoint;
  for (final cmd in commands) {
    if (cmd == 'HEAT') {
      mode = 'HEAT';
    } else if (cmd == 'COOL') {
      mode = 'COOL';
    } else if (cmd == 'OFF') {
      mode = 'OFF';
    } else if (cmd.startsWith('SET ')) {
      final val = double.tryParse(cmd.substring(4));
      if (val != null) {
        setpoint = val;
        if (mode == 'OFF') mode = 'HEAT';
      }
    } else if (cmd.startsWith('ADJ ')) {
      final parts = cmd.substring(4).split(' ');
      if (parts.length == 2) {
        final dir = parts[0];
        final delta = double.tryParse(parts[1]);
        if (delta != null && (dir == 'UP' || dir == 'DOWN') && mode != 'OFF') {
          if (dir == 'UP') {
            setpoint += delta;
          } else {
            setpoint -= delta;
          }
        }
      }
    }
  }
  return '$mode ${setpoint.toStringAsFixed(1)}';
}

@pragma('vm:entry-point')
void main() {
  assert(executeThermostatCommands([], 20.0) == 'OFF 20.0');
  assert(executeThermostatCommands(['HEAT'], 22.5) == 'HEAT 22.5');
  assert(executeThermostatCommands(['SET 23.0'], 18.0) == 'HEAT 23.0');
  print('All tests passed!');
}