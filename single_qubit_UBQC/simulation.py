#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:29:40 2019

@author: VeriQloud

This simulator works as follows:
There is a server and a client.
The server keeps the full representation of the state using qutip. For this purpose, he gets the hiding parameters from the client.
Server and Client both run in a different process but communicate using Pipe.

There are two qubits on the server of which one can be remotely projected by the client.

Client and Server have standard functions that can be used in the separately defined protocols.

The goal of the protocol below is to peform an arbitrary single-qubit computation by interpreting the two qubits as a one-dimensional, continuously rebuilding cluster state.
"""

from cliserv import Server,Client
import qutip as qt, numpy as np
from numpy import pi

from multiprocessing import Process, Pipe


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

    #print("S: final state:\n", server.qubit1.full())
    server.conn.send(server.qubit1)



def client_protocol(client):

    def compute_delta(angle, theta, m, r1, r2, r3, s1, s2): # this is the angle the server will have to measure at
        return ((-1)**(1+s2+r2)*angle + theta + (m+s1+r1+r3)*pi)%(2*pi)

    angle = np.random.randint(0, 3, 4)*pi/4     # these are the actual angles only known to the client; the set {0, pi/4, pi/2} is sufficient for universal computation
    angle[3] = 0    # this will just do a Hadamard at the end
    theta = np.random.randint(0, 8, 5)*pi/8     # the hiding angles
    r = np.random.randint(0, 2, 4)              # for hiding the measurement outcome
    s1 = 0
    s2 = 0
    r1 = 0
    r2 = 0
    r3 = r[0]
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



    clean_state = qt.rz(angle[2]) * qt.rx(angle[1]) * qt.rz(angle[0]) * 1./np.sqrt(2)*qt.Qobj([[1],[1]])
    expected_state = qt.rz(theta[4]) * qt.rx(pi*(s2+r[3])) * qt.rz(pi*(m+s1+r[2])) *  qt.rz(angle[2]) * qt.rx(angle[1]) * qt.rz(angle[0]) * 1./np.sqrt(2)*qt.Qobj([[1],[1]])
    server_state = client.conn.recv()

    print("--------------------")
    print("C: final state info: \nprep_state correction m = {}, hidden angle : theta_5 =  {:.4f}, dependancies : s4 = {} s3 = {},  hidden signaling : r4 = {} r3 = {}".format(m,theta[4],s2,s1,r[3],r[2]))
    #print("|psi_out> = R_Z(theta_5) X**(s_4+r_4) Z**(m+s_3+r_3) R_Z(alpha) R_X(beta) R_Z(gamma) |psi_in>")
    print("non-hidden state:\n", clean_state.full())
    print("state on the server side:\n", server_state.full())
    print("state should be:\n", expected_state.full())
    print("overlap: ", abs(expected_state.overlap(server_state)))




if __name__ == "__main__":

    parent_conn, child_conn = Pipe()
    S = Server(parent_conn)
    C = Client(child_conn)



    p1 = Process(target=server_protocol, args=(S,))
    p2 = Process(target=client_protocol, args=(C,))

    p1.start()
    p2.start()
