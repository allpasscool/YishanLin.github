import socket
import sys

HOST = sys.argv[1]    # The remote host
PORT = 80              # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

##############starts with http:// ????
if(HOST.startswith("http://") == False):
    #print("wrong starts")
    sys.exit(7)

##############if https, can't handle!!
if(HOST.split("://")[0] == "https"):
    #print("https address, can't handle. exit!")
    sys.stderr.write("https address, can't handle. exit! stderr\n")
    sys.exit(1)
##############set address to acceptable format
HOST = HOST.split("://")[1]
if(len(HOST.split("/")) > 1):
    path = HOST.split("/")[1]
else:
    path = ""
if(len(HOST.split("/")[0].split(":")) > 1):
    PORT = int(HOST.split("/")[0].split(":")[1].split("/")[0])
HOST = HOST.split("/")[0].split(":")[0]

#print("after split, " + HOST)

#print("after socket")
##############connect
try:
    s.connect((HOST, PORT))
except socket.error as msg:
    #print("get wrong!")
    #print(msg)
    sys.exit(2)
#print("after connet")

#print("before send")
##############send
s.sendall("GET /" + path +" HTTP/1.0\r\nHost: " + HOST + ":" + str(PORT) + "\r\n\r\n")

#print("after send")
#i = 0
countOfRedirect = 0

##############reading package
while True:
    ##############receive data
    resp = s.recv(1024)
    while True:
        NextResp = s.recv(1024)
        if(NextResp == ""):
            break
        resp +=  NextResp
    #print("line 52 Resp:%s" %(resp))
    #redirect to another address
    #print("before resp")
    #print("resp  split " + resp.split("\r\n")[0])
    statusCode = resp.split("\n")[0].split(" ")[1]
    #print("after calculate status code:" + statusCode)
    ##############status code >= 400
    if(int(statusCode) >= 400):
        #print("status code: " + statusCode)
        if ("Content-Type: text/html" in resp):
            if ("<body" in resp and "</body>" in resp):
                print("<body" + resp.split("<body")[1].split("</body>")[0] + "</body>")
            elif ("<body" in resp and "</BODY>" in resp):
                print("<body" + resp.split("<body")[1].split("</BODY>")[0] + "</BODY>")
            elif ("<BODY" in resp and "</body>" in resp):
                print("<BODY" + resp.split("<BODY")[1].split("</body>")[0] + "</body>")
            else:
                print("<BODY" + resp.split("<BODY")[1].split("</BODY>")[0] + "</BODY>")
        #print("after")
        s.close()
        sys.exit(5)
    ##############if redirect
    if(("HTTP/1.1 301" in resp) or ("HTTP/1.1 302" in resp) or
            ("HTTP/1.0 301" in resp) or ("HTTP/1.0 302" in resp)):
        if ("Content-Type: text/html" in resp):
            if("<body" in resp and "</body>" in resp):
                print("<body" + resp.split("<body")[1].split("</body>")[0] + "</body>")
            elif("<body" in resp and "</BODY>" in resp):
                print("<body" + resp.split("<body")[1].split("</BODY>")[0] + "</BODY>")
            elif("<BODY" in resp and "</body>" in resp):
                print("<BODY" + resp.split("<BODY")[1].split("</body>")[0] + "</body>")
            else:
                print("<BODY" + resp.split("<BODY")[1].split("</BODY>")[0] + "</BODY>")
        countOfRedirect += 1
        newLocation = resp.split("Location: ")[1].split("\r\n")[0]
        #print("redirect to " + newLocation)
        sys.stderr.write("Redirected to: " + newLocation + "\n")
        HOST = newLocation
        #print(HOST)
        #print(HOST.split("://")[1])
        ############## if https
        if (HOST.split("://")[0] == "https"):
            #print("https address, can't handle. exit!")
            sys.stderr.write("https address, can't handle. exit! stderr\n")
            s.close()
            sys.exit(1)
        ##############change host and path format
        HOST = HOST.split("://")[1]
        if(len(HOST.split("/")) > 1):
            path = HOST.split("/")[1].split("\r\n")[0]
        else:
            path = ""
        HOST = HOST.split("/")[0].split(":")[0]
        if (len(HOST.split("/")[0].split(":")) > 1):
            PORT = int(HOST.split("/")[0].split(":")[1].split("/")[0])

        #print("after redirect and after split, " + HOST + "  path: " + path)
        ##############shut down socket
        s.shutdown(socket.SHUT_RDWR)
        s.close()
        #print("after close socket")
        ##############create a new socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #print("after recreate socket")
        s.connect((HOST, PORT))

        #print("before send, Host: %s path: %s " % (HOST, path))
        ##############send data
        s.sendall("GET /" + path + " HTTP/1.0\r\nHost: " + HOST + ":" + str(PORT) + "\r\n\r\n")
        if(countOfRedirect >= 10):
            s.close()
            sys.exit(6)
        continue
    #print(i)
    if ("Content-Type: text/html" in resp):
        if ("<body" in resp and "</body>" in resp):
            print("<body" + resp.split("<body")[1].split("</body>")[0] + "</body>")
        elif ("<body" in resp and "</BODY>" in resp):
            print("<body" + resp.split("<body")[1].split("</BODY>")[0] + "</BODY>")
        elif ("<BODY" in resp and "</body>" in resp):
            print("<BODY" + resp.split("<BODY")[1].split("</body>")[0] + "</body>")
        else:
            print("<BODY" + resp.split("<BODY")[1].split("</BODY>")[0] + "</BODY>")
    s.close()
    sys.exit(0)