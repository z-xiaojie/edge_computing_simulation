from Client import Helper
import socket
import sys

if __name__ == '__main__':
    # 192.168.1.162 [0, 1, 2, 3, 4]
    if len(sys.argv) >= 3:
        helper = Helper(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        helper = Helper("192.168.1.162", 0, 3)
    print("helper")
    while True:
        try:
            helper.connect()
            helper.optimize()
        except socket.error:
            print("no request, waiting...")

