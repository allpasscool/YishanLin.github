import socket
import sys
import os.path

HOST = ''             # Symbolic name meaning all available interfaces
PORT = int(sys.argv[1])    # The port number

####address families are AF_INET (IP), AF_INET6 (IPv6), AF_UNIX (local channel, similar to pipes)
#### , AF_ISO (ISO protocols), and AF_NS (Xerox Network Systems protocols)
####type of service. SOCK_STREAM (virtual circuit service), SOCK_DGRAM (datagram service),
####  SOCK_RAW (direct IP service)
####protocol
####indicate a specific protocol to use in supporting the sockets operation.
####  This is useful in cases where some families may have more than one protocol
####  to support a given type of service. The return value is a file descriptor
#### (a small integer). The analogy of creating a socket is that of requesting
####  a telephone line from the phone company.
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
####Listen for connections made to the socket.
####  The backlog argument specifies the maximum number of queued connections
####  and should be at least 0; the maximum value is system-dependent (usually 5),
####  the minimum value is forced to 0.
s.listen(5)

while True:
    conn, addr = s.accept() # Establish connection with client.
    print 'Connected by', addr
    print "connection:", conn

    data = conn.recv(1024)
    try:
        #while True:
        #    nextData = conn.recv(1024)
        #    print("Next data:%s" %nextData)
        #    if nextData == "":
        #        break
        #    data += nextData
        #print(data)
        #print(data.split("\r\n")[0].split(" ")[0])
        if("GET" in data.split("\r\n")[0].split(" ")[0]):
            fileName = data.split("\r\n")[0].split(" ")[1]
            #print("client request:%s" %fileName)
            #print("If file exists: ")
            #print(os.path.isfile("." + fileName))
            if(os.path.isfile("." + fileName) and (fileName.endswith("htm") or fileName.endswith("html"))):
                #print("html file exists")
                conn.sendall("HTTP/1.0 200 OK\nContent-Type: text/html\n")
                conn.sendall("Content-Type: text/html\n")
                webFile = open(fileName.split("/")[1], "r")
                if webFile.mode == 'r':
                    #print("time for contents")
                    while True:
                        contents = webFile.readline()
                        #print(contents)
                        if(contents == ""):
                            break
                        else:
                            conn.sendall(contents)
                    #print("ready to shutdown")
                conn.close()
            #elif file doesn't exist
            elif(not os.path.isfile("." + fileName)):
                conn.sendall("HTTP/1.0 404 Not Found\n")
                #print("404")
            #elis file doesn't end up with htm or html
            elif(os.path.exists("." + fileName) and (not fileName.endswith("htm") and not fileName.endswith("html"))):
                conn.sendall("HTTP/1.0 403 Forbidden\n")
                #print("403")
            else:
                #print("something wrong!!")
                conn.close()
    except:
        conn.close()
        #print("try again")

s.close()