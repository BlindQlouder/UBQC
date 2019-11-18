"""
Created on Tue Apr 24 16:40:00 2019

@author: VeriQloud

The core engine for simulating the client and the server
"""


import qutip as qt, random, time, numpy as np
from numpy import pi

from multiprocessing import Process, Pipe, Queue

class Server():
    def __init__(self, conn, queue):
        self.qubit1 = qt.Qobj([[1],[0]])    # the computational qubit
        self.qubit2 = qt.Qobj([[1],[0]])    # the qubit the client can prepare remotely; measurements are done exclusively on this qubit
        self.system = qt.tensor(self.qubit1,self.qubit2)
        self.conn = conn
        self.queue = queue

    def RS(self):   # remote state preparation
        print("S: applying RS")
        theta = self.conn.recv()    # this is the hiding angle of the client. Normally, the client would keep it for himself.
        self.qubit2 = qt.rz(theta)*(qt.snot()*qt.Qobj([[1],[0]])) # this is |0> + exp(i*theta)|1>
        self.system = qt.tensor(self.qubit1,self.qubit2)
        return 1

    def SWAP(self):
        print("S: applying SWAP")
        self.system = qt.swap()*self.system
        return 1

    def CPHASE(self):
        print("S: applying CPHASE")
        self.system = qt.cphase(pi)*self.system
        return 1

    def get_angle(self):
        print("S: getting angle")
        return self.conn.recv()

    def U_client(self, delta):
        print("S: applying U_client")
        self.system = qt.tensor(qt.qeye(2), qt.snot()*qt.rz(-delta)) * self.system            # rotation around Z followed by Hadamard on the second qubit
        return 1

    def M(self):    # measurement in standard basis on the second qubit
        print("S: applying M")
        psi = self.system.full().flatten()
        p = np.abs(psi)**2
        if random.random()<=(p[0] + p[2]):
            m = 0
            q1 = qt.Qobj([[psi[0]], [psi[2]]])
        else:
            m = 1
            q1 = qt.Qobj([[psi[1]], [psi[3]]])
        q1 = q1/q1.norm()
        self.qubit1 = q1
        #print("M: qubit1:", self.qubit1)
        return m

    def send_measurement(self, m):
        print("S: sending measurement")
        self.conn.send(m)
        return 1


class Client():
    def __init__(self, conn):
        self.conn = conn

    def RS(self, theta):    # projection onto a basis { |0> + exp(i*theta)|1>, |0> - exp(i*theta)|1> }
        print("C: applying RS")
        m = random.randint(0,1) # measurement outcome
        self.conn.send(theta+m*pi)
        return m

    def send_angle(self, delta):
        print("C: sending angle")
        self.conn.send(delta)
        return 1

    def get_measurement(self):
        m = self.conn.recv()
        return m

def StartSimulation(server_process, client_process):
    parent_conn, child_conn = Pipe()
    queue = Queue()
    S = Server(parent_conn, queue)
    C = Client(child_conn)

    p1 = Process(target=server_process, args=(S,))
    p2 = Process(target=client_process, args=(C,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    


    states = queue.get()

    return states