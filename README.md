# UBQC (Universal Blind Quantum Computation)

Currently this repository contains a single file for simulating single-qubit UBQC.

We simulate a client-server setting where the server has two qubits. The server has write only qubits, which means it interacts with the client by emitting photons that are maximally entangled with a solid qubit. The client can then prepares one of the qubits by measuring the photon, which projects entangled qubits onto a chosen basis.

Since the angle of the basis chosen by the client is unknown to the server, this allows the client to hide the computation. After the remote preparation, the client intstructs the server to perform operations on the two qubits. This can be repeated as required. We demonstrate how to use it with a protocol whose final result is an arbitrary state on the server that is known to the client but appears completely random from the server's perspective.
