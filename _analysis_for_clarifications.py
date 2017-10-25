import numpy as np
import matplotlib.pyplot as plt
import cantilevers
import materials
import finite_element
import gaussian


def main():
    
    material = materials.PiezoMumpsMaterial()
    neta1, neta2, k1, k2 = [], [], [], []
    for l in range(40, 81):
        print('Length %d' % l)
        c1 = cantilevers.RectangularCantilever(5e-6, 5e-6, 20, l)
        c2 = cantilevers.RectangularCantilever(5e-6, 5e-6, 20, 80)
        fem1 = finite_element.LaminateFEM(c1, material)
        fem2 = finite_element.LaminateFEM(c2, material)
        w1, v1 = fem1.modal_analysis(1)
        w2, v2 = fem2.modal_analysis(1)
        gau1 = gaussian.Gaussian(fem1.get_mesh(), c1.xtip, c1.ytip_ub, 0.1)
        guu1 = gau1.get_operator()
        gau2 = gaussian.Gaussian(fem2.get_mesh(), c1.xtip, c1.ytip_ub, 0.1)
        guu2 = gau2.get_operator()
        kuu1 = fem1.get_stiffness_matrix()
        kuv1 = fem1.get_piezoelectric_matrix()
        kuu2 = fem2.get_stiffness_matrix()
        kuv2 = fem2.get_piezoelectric_matrix()
        
        phi1 = v1[:, [0]]
        phi2 = v2[:, [0]]
        wtip1 = np.asscalar(guu1 @ phi1)
        wtip2 = np.asscalar(guu2 @ phi2)
        charge1 = -1e6 * np.asscalar(kuv1.T @ phi1)
        charge2 = -1e6 * np.asscalar(kuv2.T @ phi2)
        neta1.append(charge1 / wtip1)
        neta2.append(charge2 / wtip2)
        k1.append(np.asscalar(phi1.T @ kuu1 @ phi1 / wtip1 ** 2))
        k2.append(np.asscalar(phi2.T @ kuu2 @ phi2 / wtip2 ** 2))
    return neta1, neta2, k1, k2
    

if __name__ == '__main__':
    neta1, neta2, k1, k2 = main()
    fig, ax = plt.subplots()
    ax.plot(k1, neta1, label='Change in Length')
    ax.plot(k2, neta2, label='Change in Tip')
    ax.legend()
    plt.show()
            