import socket
ports = [20, 21, 22, 23, 25, 53, 80, 110, 123, 443, 1433, 3306, 1521, 8080, 3389]
host = socket.gethostname()
for port in ports:
    try:
        s = socket.socket()
        s.connect((host, port))
        print('オープンされているポート:%d' % port)
        s.close()

    except: pass