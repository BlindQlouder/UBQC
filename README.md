# UBQC (Universal Blind Quantum Computation)

Currently this repository contains a single file for simulating single-qubt UBQC.

We assume a client server setting where the Server has two qubits and the client can remotely prepare one of the qubits by projecting it onto a rotated basis. The angle of the basis is unknown to the server and allows the client to hide the computation. After the remote preparation, the client intstructs the server to perform operations on the two qubits. This can be repeated as required. The final result is an arbitrary state on the server that is known to the client but appears completely random from the server's perspective. 