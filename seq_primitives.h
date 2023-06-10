

void seqSubstractMat(int n, int m, float *A, float *B, float *C);

void seqTranspose(int n, int m, float *A, float *B);

void seqMatMult(int n, int m, int p, float *A, float *B, float *C);

void seqDerivativeReLu(int n, int m, float *A, float *B);

void seqElementWiseProduct(int n, int m, float *A, float *B, float *C);

void seqScalarDivMat(int n, int m, float *A, float scalar, float *B);

void seqScalarProdMat(int n, int m, float *A, float scalar, float *B);

void seqSoftmax(int nOutput, int batchSize, float *A, float *B);

void seqReLu(int n, int m, float *A, float *B);

void seqCrossEntropy(int batchSize, int nOutput, float *h_result, float *h_labels, float *h_loss);

float seqAccuracy(int batchSize, int nOutput, float *h_result, float *h_labels);

int seqMaxIndex(int n, float *A);