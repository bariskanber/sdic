# SDIC

Sparse matrix to structured imageset conversion

Converts an M by N sparse matrix to M images of P by P while attempting to give each image a structure that is amenable for use with convolutional neural networks

P is given by the smallest integer that is divisible by 2 that is greater than or equal to sqrt(N)
