#define MASK 0xffffffff

#define ROLL16(x) x = (((x) << 16) | ((x) >> 16)) & MASK
#define ROLL12(x) x = (((x) << 12) | ((x) >> 20)) & MASK
#define ROLL8(x) x = (((x) << 8) | ((x) >> 24)) & MASK
#define ROLL7(x) x = (((x) << 7) | ((x) >> 25)) & MASK

#define QR(x, ind, a, b, c, d)\
    x[ind][a] += x[ind][b];\
    x[ind][a] &= MASK;\
    x[ind][d] ^= x[ind][a];\
    ROLL16(x[ind][d]);\
    x[ind][c] += x[ind][d];\
    x[ind][c] &= MASK;\
    x[ind][b] ^= x[ind][c];\
    ROLL12(x[ind][b]);\
    x[ind][a] += x[ind][b];\
    x[ind][a] &= MASK;\
    x[ind][d] ^= x[ind][a];\
    ROLL8(x[ind][d]);\
    x[ind][c] += x[ind][d];\
    x[ind][c] &= MASK;\
    x[ind][b] ^= x[ind][c];\
    ROLL7(x[ind][b])
    
#define ONE_ROUND(x, ind)\
    QR(x, ind, 0, 4,  8, 12);\
    QR(x, ind, 1, 5,  9, 13);\
    QR(x, ind, 2, 6, 10, 14);\
    QR(x, ind, 3, 7, 11, 15);\
    QR(x, ind, 0, 5, 10, 15);\
    QR(x, ind, 1, 6, 11, 12);\
    QR(x, ind, 2, 7,  8, 13);\
    QR(x, ind, 3, 4,  9, 14)
