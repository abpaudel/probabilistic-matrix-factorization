import numpy as np
import sys


def PMF(train_data):
    lam = 2
    sigma2 = 0.1
    d = 5
    maxiter = 50
    Nu = int(np.amax(train_data[:,0]))
    Nv = int(np.amax(train_data[:,1]))

    v = np.zeros((Nv, d))
    for j in range(Nv):
        v[j] = np.random.normal(0, 1/float(lam), d)

    u = np.zeros((Nu, d))

    omega_u = []
    for i in range(Nu):
        omega_u.append(train_data[train_data[:,0]==i+1][:,1].astype(np.int64))

    omega_v = []
    for j in range(Nv):
        omega_v.append(train_data[train_data[:,1]==j+1][:,0].astype(np.int64))

    M = np.zeros((Nu, Nv))
    for data in train_data:
        M[int(data[0])-1, int(data[1])-1] = data[2]

    L = []
    U = []
    V = []

    for itr in range(maxiter):
        for i in range(Nu):
            vj = v[omega_u[i]-1]
            t1 = lam*sigma2*np.eye(d) + np.dot(vj.T, vj)
            t2 = (vj*M[i, omega_u[i]-1][:,None]).sum(axis=0)
            u[i] = np.dot(np.linalg.inv(t1), t2)

        for j in range(Nv):
            ui = u[omega_v[j]-1]
            t1 = lam*sigma2*np.eye(d) + np.dot(ui.T, ui)
            t2 = (ui*M[omega_v[j]-1, j][:,None]).sum(axis=0)
            v[j] = np.dot(np.linalg.inv(t1), t2)
        
        t1 = 0
        for val in train_data:
            i = int(val[0])
            j = int(val[1])
            t1 = t1 + (val[2] - np.dot(u[i-1,:],v[j-1,:]))**2
        t1 = t1/(2*sigma2)
        t2 = lam*0.5*(((np.linalg.norm(u, axis=1))**2).sum())
        t3 = lam*0.5*(((np.linalg.norm(v, axis=1))**2).sum())
        l = -t1-t2-t3
        
        L.append(l)
        U.append(u)
        V.append(v)

    return L, U, V


def main():
    train_data = np.genfromtxt(sys.argv[1], delimiter = ",")
    L, U_matrices, V_matrices = PMF(train_data)

    np.savetxt("objective.csv", L, delimiter=",")

    np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
    np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
    np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

    np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
    np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
    np.savetxt("V-50.csv", V_matrices[49], delimiter=",")


if __name__ == "__main__":
    main()