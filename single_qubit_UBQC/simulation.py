#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:29:40 2019

@author: BlindQlouder

This simulator works as follows:
There is a server and a client. 
The server keeps the full representation of the state using qutip. For this purpose, he has the hiding paramenters from the client, which would be hidden in a real implementation. 
Server and Client both run in a different process but communicate using Pipe.

The goal of the protocol is to have a qubit roated to an arbitrary position. 
"""

import qutip as qt, random, time, numpy as np
from numpy import pi
from multiprocessing import Process, Pipe



class Server():
    def __init__(self, conn):
        self.qubit1 = qt.Qobj([[1],[0]])
        self.qubit2 = qt.Qobj([[1],[0]])
        self.system = qt.tensor(self.qubit1,self.qubit2)
        self.conn = conn

    def RS(self):   # remote state preparation
        print("S: applying RS")
        theta = self.conn.recv()    # this is the hiding angle of the client. Normally, the client would keep it for himself.
        self.qubit2 = qt.rz(theta)*(qt.snot()*qt.Qobj([[1],[0]]))
        self.system = qt.tensor(self.qubit1,self.qubit2)
        #print(self.system)
        return 1

    def SWAP(self):
        print("S: applying SWAP")
        self.system = qt.swap()*self.system
        #print(self.system)
        return 1

    def CPHASE(self):
        print("S: applying CPHASE")
        self.system = qt.cphase(pi)*self.system
        #print(self.system)
        return 1

    def get_angle(self):
        print("S: getting angle")
        return self.conn.recv()

    def U_client(self, delta):
        print("S: applying U_client")
        self.system = qt.tensor(qt.qeye(2), qt.snot()*qt.rz(-delta)) * self.system            # rotation around Z followed by Hadamard on the second qubit
        #print(self.system)
        return 1

    def M(self):
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
        print("M: qubit1:", self.qubit1)
        #self.conn.send(m)
        return m
        
    def send_measurement(self, m):
        print("S: sending measurement")
        self.conn.send(m)
        return 1
        





class Client():
    def __init__(self, conn):
        self.conn = conn
        print("client created")

    def RS(self, theta):
        print("C: applying RS")
        m = random.randint(0,1)
        self.conn.send(theta+m*pi)
        return m

    def send_angle(self, delta):
        print("C: sending angle")
        self.conn.send(delta)
        return 1

    def get_measurement(self):
        m = self.conn.recv()
        return m





if __name__ == "__main__":

    parent_conn, child_conn = Pipe()
    S = Server(parent_conn)
    C = Client(child_conn)

    def server_protocol(server):
        server.RS()
        server.SWAP()   
        m = server.M()  

        for i in range(1,5):
            delta = server.get_angle()
            server.RS()
            server.SWAP()
            server.CPHASE()
            server.U_client(delta)
            m = server.M()
            server.send_measurement(m)

        #recv a last measurement requirement ? 
        #delta = server.get_angle()
        #SWAP (or do the single qubit last measurement on memory ion ?)
        #m = server.U_client(delta)
        #server.send_measurement(m)
        print("----------")
        print("S: final state:\n", server.qubit1.full())
        print("----------")




    def client_protocol(client):

        def compute_delta(angle, theta, m, r1, r2, r3, s1, s2):
            return ((-1)**(1+s2+r2)*angle + theta + (m+s1+r1+r3)*pi)%(2*pi)

        alpha = pi/4
        beta = pi/2
        gamma = -pi/2
        angle = [gamma, beta, alpha, 0]
        angle = [0, beta, 0, 0]
        theta = np.random.randint(0, 8, 5)*pi/8
        #theta = [0, 0, 0, 0, 0]
        theta[4] = 0
        r = np.random.randint(0, 2, 4)
        s1 = 0
        s2 = 0
        r1 = 0
        r2 = 0
        r3 = r[0]
        # here |psi_in> = |+>. Choose RS(in_angle+theta1) for in_angle not zero(ex : classical input : |psi_in> = |+> or |->), for (any) quantum input, (second modulator is available ?) one time pad it with X1**c, c in {0,1} 
        m = client.RS(theta[0])
        delta = compute_delta(angle[0], theta[0], m, r1, r2, r3, s1, s2)      


        for i in range(1,5):
            client.send_angle(delta)
            m = client.RS(theta[i])
            s1 = s2
            s2 = client.get_measurement()   
            if i < 4: 
                r1 = r2
                r2 = r3
                r3 = r[i]
                delta = compute_delta(angle[i], theta[i], m, r1, r2, r3, s1, s2)

        #choose a last measurement (tomography ?) on qubit 5 supposed to be in the state X**(s_4+r_4).Z**(s_3+r_3).R_Z(theta_5)R_Z(alpha)R_X(beta)R_Z(gamma)|psi_in> ? 
        #print("----------")
        time.sleep(0.01)
        print("C: final state info: \nprep_state correction m = {}, hidden angle : theta_5 =  {:.4f}, dependancies : s4 = {} s3 = {},  hidden signaling : r4 = {} r3 = {}".format(m,theta[4],s2,s1,r[3],r[2]))
        print("|psi_out> = X**(s_4+r_4) Z**(m+s_3+r_3) R_Z(theta_5) R_Z(alpha) R_X(beta) R_Z(gamma) |psi_in>")
        print("|psi_in> = |+>, m+s_3+r_3 = {}".format((m+s1+r[2])%2))
        #print("----------")
        expected_state = qt.rx(pi*(s2+r[3])) * qt.rz(pi*(m+s1+r[2])) * qt.rz(theta[4]) * qt.rz(angle[2]) * qt.rx(angle[1]) * qt.rz(angle[0]) * 1./np.sqrt(2)*qt.Qobj([[1],[1]])
        psi = expected_state.full()
        print("state should be:\n", psi)


    p1 = Process(target=server_protocol, args=(S,))
    p2 = Process(target=client_protocol, args=(C,))

    p1.start()
    p2.start()

    

