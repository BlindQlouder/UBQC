# UBQC (Universal Blind Quantum Computation)

Currently this repository contains files for simulating single-qubit UBQC.

We simulate a client-server setting where the server has two qubits: a communication qubit and a memory qubit. The server's qubits are write-only, which means it interacts with the client by emitting photons that are maximally entangled with the communication qubit. The client then prepares it by measuring the photon, which projects the entangled communication qubits onto a chosen basis. It can then be swapped with the memory qubit which has a longer coherence time.

Since the angles of the measurement bases chosen by the client are unknown to the server, this allows the client to hide the computation. After the remote preparation, the client intstructs the server to perform operations on the two qubits. This can be repeated as required. 

We demonstrate how to use this procedure with a protocol whose final result is an arbitrary state on the server that is known to the client but appears completely random from the server's perspective.
