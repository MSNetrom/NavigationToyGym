import numpy as np
import scipy as sp
from abc import ABC, abstractmethod

def single_exponential_cbf_solver(u_ref: np.ndarray, Lg_psi: np.ndarray, Lf_psi: float, psi: float, p1: float) -> np.ndarray:

    alpha = Lg_psi
    beta = - Lf_psi - p1 * psi

    a_len_squared = np.sum(alpha ** 2)

    # If a_len_squared is (almost) zero, then return u_ref
    if a_len_squared < 1e-6:
        return u_ref
    
    # Else do the derived solution
    return max(beta - np.dot(alpha, u_ref), 0) * alpha / a_len_squared + u_ref

class FirstOrderGeneralLie(ABC):

    @abstractmethod
    def get_Lg_Lf_and_psi(self, states: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


class SoftMinLie(FirstOrderGeneralLie):

    def __init__(self, first_order_cbfs: list[FirstOrderGeneralLie], k: float):
        self.first_order_cbfs = first_order_cbfs
        self.k = k

        self.h_track = []
        self.psi_track = []

    def get_h_and_psi(self):
        return self.h_track, self.psi_track

    def get_Lg_Lf_and_psi(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        original_Lg_psis, original_Lf_psis, original_psis, original_hs = zip(*[cbf.get_Lg_Lf_and_psi(states) for cbf in self.first_order_cbfs])
        # Concatenate all CBF components
        Lg_psis = np.concatenate([original_Lg_psi for original_Lg_psi in original_Lg_psis], axis=0)
        Lf_psis = np.concatenate([original_Lf_psi for original_Lf_psi in original_Lf_psis], axis=0)
        psis = np.concatenate([original_psi for original_psi in original_psis], axis=0)
        hs = np.concatenate([original_h for original_h in original_hs], axis=0)

        self.h_track.append(hs.min(axis=0))
        self.psi_track.append(psis.min(axis=0))

        #return Lg_psis, - Lf_psis - self.p1 * psis

        # Compute the softmin as defined in the paper
        ins = -self.k * psis
        logsumexp_val = sp.special.logsumexp(ins)
        combined_psi = - logsumexp_val / self.k

        # Correct coefficient calculation using logsumexp for numerical stability
        coeffs = np.exp(ins - logsumexp_val)

        # Compute combined Lf and Lg
        combined_Lf_psi = np.sum(coeffs * Lf_psis, axis=0)
        combined_Lg_psi = np.sum(coeffs.reshape(-1, 1) * Lg_psis, axis=0)

        return combined_Lg_psi, combined_Lf_psi, combined_psi