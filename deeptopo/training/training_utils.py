# Recherche du biais optimal par dichtomie
def root_finder(xhat, vt, func):
    bmin, bmax = -1.e7, 1.e7

    while (bmax - bmin)/(abs(bmax) + abs(bmin)) > 1.e-5:
        bmid = 0.5*(bmin + bmax)
        vmid = func(bmid + xhat).sum()
        if vmid > vt:
            bmax = bmid
        else:
            bmin = bmid
    return bmid
