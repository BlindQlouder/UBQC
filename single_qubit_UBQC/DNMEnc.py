#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:20:00 2019

@author: VeriQloud

This simulates the following client-server interaction:
The server prepares a linear cluster state in which each state is prepared via steering
by the client. The client then sends for each qubit the measurements to be performed
by the server. Since the server only has 2 qubits, each the i-th state in the cluster
is created after state (i-2) has been consumed/measured.

The resulting state is a random state, sampled from a 2-design.
"""

from cliserv import StartSimulation
import qutip as qt, numpy as np, sys, os
from numpy import pi
import matplotlib.pylab as plt

N=6 #The length of the linear cluster

num_runs = 1   # repeat runs to get statistics

def server_protocol(server):
    result = []
    for run in range(num_runs):
        #Preparing the initial state
        server.RS()
        server.SWAP()
        m = server.M()

        for i in range(1,N):
            # The server receives the measurement angle for that round from the client
            delta = server.get_angle()
            # The server gets the state through RSP, swaps and entangles it
            server.RS()
            server.SWAP()
            server.CPHASE()

            # The server applies the correct measurement sends the result to the client
            server.U_client(delta)
            m = server.M()
            server.send_measurement(m)

        print("S: final state:\n", server.qubit1.full())
        #print("S: final state:\n", np.abs(server.qubit1.full().flatten())**2)
        result.append(server.qubit1.full().flatten())
    server.queue.put(np.array(result))

def client_protocol(client):

    def compute_delta(phi, theta, r1, r2, r3, s):
        """
        This function computes the measurement communicating by the client to the server, hiding its description
        Parameters:
        phi: the angle of the state on the server
        theta: the angle of the targeted measurement
        r1, r2, r3: the classical hiding for qubits i-2, i-1, and i
        s: the measurement's outcomes for qubits i-1
        """
        return ((-1)**(s+r2)*phi + theta + (r1+r3)*pi)%(2*pi)


    for run in range(num_runs):

        # The angles for the qubits in the server's cluster state, prepared through RSP
        # chosen in {pi/4, 3pi/4, 5pi/4, 7pi/4}
        phi = (2*np.random.randint(0, 3, N)+1)*pi/4
        #phi = np.ones(N) * pi/4
        #phi[3] = 0    # this will just do a Hadamard at the end #We should do it explicitely if required and not via a trick

        # The hiding angles chosen by the client and sent to the server
        # chosen from {0, pi/2, pi, 3pi/2}
        theta = np.random.randint(0, 3, N)*pi/2

        # Randomness for hiding the measurement's outcomes
        r = np.random.randint(0, 2, N-1)

        #First round:
        s = 0
        r1 = 0
        r2 = 0
        r3 = r[0]
        m = client.RS(theta[0])
        # The description of the state is updated based on the result of the measurement
        phi[0] = (-1)**m*phi[0]
        delta = compute_delta(phi[0], theta[0], 0, 0, r[0], s)

        for i in range(1,N):
            client.send_angle(delta)
            m = client.RS(theta[i])
            # The description of the state is updated based on the result of the measurement
            phi[i] = (-1)**m*phi[i]
            s = client.get_measurement()
            if i < N-1:
                r1 = r2
                r2 = r3
                r3 = r[i]
                delta = compute_delta(phi[i], theta[i], r1, r2, r3, s)


        #clean_state = qt.rz(angle[2]) * qt.rx(angle[1]) * qt.rz(angle[0]) * 1./np.sqrt(2)*qt.Qobj([[1],[1]])
        #expected_state = qt.rz(theta[4]) * qt.rx(pi*(s2+r[3])) * qt.rz(pi*(m+s1+r[2])) *  qt.rz(angle[2]) * qt.rx(angle[1]) * qt.rz(angle[0]) * 1./np.sqrt(2)*qt.Qobj([[1],[1]])

        #print("--------------------")
        #print("C: final state info: \nprep_state correction m = {}, hidden angle : theta_5 =  {:.4f}, dependancies : s4 = {} s3 = {},  hidden signaling : r4 = {} r3 = {}".format(m,theta[4],s2,s1,r[3],r[2]))
        #print("|psi_out> = R_Z(theta_5) X**(s_4+r_4) Z**(m+s_3+r_3) R_Z(alpha) R_X(beta) R_Z(gamma) |psi_in>")
        #print("non-hidden state:\n", clean_state.full())
        #print("state on the server side:\n", server_state.full())
        #print("state should be:\n", expected_state.full())
        #print("overlap: ", abs(expected_state.overlap(server_state)))



def plot_bloch_sphere(states):
    b = qt.Bloch()
    l = [qt.Qobj(state) for state in states]
    b.add_states(l)
    b.show()

def plot_statistics(states):
    print('plotting histogram...')
    data = []
    for state in states:
        #rho = state * state.dag()
        w = np.abs(state[0])**2 - np.abs(state[1])**2
        u = 2. * (state[0] * state[1].conj()).real
        v = 2. * (state[1] * state[0].conj()).imag
        theta = np.arccos(w)
        if v>=0 :
            phi = np.arccos(u/np.sin(theta))
        else:
            phi = 2*np.pi - np.arccos(u/np.sin(theta))


        data.append([theta, phi])

    data = np.array(data)
    #data.reshape(data.shape[0],data.shape[1])
    h1, b1 = np.histogram(data[:,0], bins=200, range=(0,np.pi))
    h2, b2 = np.histogram(data[:,1], bins=200, range=(0,2*np.pi))
    plt.figure()
    plt.plot(b1[1:]*2, h1, label='2*theta')
    plt.plot(b2[1:], h2, label='phi')
    plt.legend()
    plt.show()

if __name__ == "__main__":

    verbose = True

    if not verbose:
        sys.stdout = open(os.devnull, 'w')
    
#    final_states = []
    #for i in range(num_runs):
#        final_states.append(StartSimulation(server_protocol, client_protocol))
    
    final_states = StartSimulation(server_protocol, client_protocol)

    sys.stdout = sys.__stdout__

    #plot_bloch_sphere(final_states)

    #plot_statistics(final_states)

    

    











