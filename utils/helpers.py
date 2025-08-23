import math


def inv_kl(qs, ks):
    """Numerically invert the binary KL for PAC-Bayes-kl bound."""
    izq = qs
    dch = 1 - 1e-10
    qd = 0
    while((dch - izq) / dch >= 1e-5):
        p = (izq + dch) * 0.5
        if qs == 0:
            ikl = ks - (0 + (1 - qs) * math.log((1 - qs) / (1 - p)))
        elif qs == 1:
            ikl = ks - (qs * math.log(qs / p) + 0)
        else:
            ikl = ks - (qs * math.log(qs / p) + (1 - qs) * math.log((1 - qs) / (1 - p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd