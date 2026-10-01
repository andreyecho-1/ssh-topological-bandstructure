import numpy as np
import matplotlib.pyplot as plt

def generate_ssh_bands(v, w, k_points=300):
    """
    Calculates the energy band structure for the 1D Su-Schrieffer-Heeger (SSH) model.
    v: Intracell hopping amplitude
    w: Intercell hopping amplitude
    k_points: Number of momentum points across the Brillouin zone
    """
    k_array = np.linspace(-np.pi, np.pi, k_points)
    energy_plus = []
    energy_minus = []

    for k in k_array:
        # Construct the 2x2 SSH Bloch Hamiltonian matrix
        # H(k) = [[0, v + w*exp(-ik)], [v + w*exp(ik), 0]]
        off_diag = v + w * np.exp(-1j * k)
        H_k = np.array([[0, off_diag], 
                        [np.conj(off_diag), 0]])
        
        # Diagonalize matrix to obtain real energy eigenvalues
        eigenvalues = np.linalg.eigvalsh(H_k)
        energy_minus.append(eigenvalues[0])
        energy_plus.append(eigenvalues[1])

    return k_array, energy_minus, energy_plus

def plot_band_structure():
    k_array, e_minus_triv, e_plus_triv = generate_ssh_bands(v=1.0, w=0.5)
    _, e_minus_topo, e_plus_topo = generate_ssh_bands(v=0.5, w=1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Trivial Phase (v > w)
    ax1.plot(k_array, e_plus_triv, 'b-', label='Conduction Band')
    ax1.plot(k_array, e_minus_triv, 'r-', label='Valence Band')
    ax1.set_title('Trivial Phase ($v = 1.0, w = 0.5$)')
    ax1.set_xlabel('Wavevector $k$')
    ax1.set_ylabel('Energy $E$')
    ax1.grid(True)
    ax1.legend()

    # Topological Phase (v < w)
    ax2.plot(k_array, e_plus_topo, 'b-', label='Conduction Band')
    ax2.plot(k_array, e_minus_topo, 'r-', label='Valence Band')
    ax2.set_title('Topological Phase ($v = 0.5, w = 1.0$)')
    ax2.set_xlabel('Wavevector $k$')
    ax2.set_ylabel('Energy $E$')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('ssh_bands.png', dpi=300)
    print("Band structure calculation complete. Image saved as ssh_bands.png.")

if __name__ == "__main__":
    plot_band_structure()
