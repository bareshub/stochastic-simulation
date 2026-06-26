"""Stochastic SIR model (chain-binomial scheme) for Project 2.

The continuous-time SIR process is simulated in discrete steps of length ``dt``.
At each step the per-individual transition probabilities are computed from the
current counts and held constant over the step,

    p_SI = 1 - exp(-beta * I / N * dt),    p_IR = 1 - exp(-gamma * dt),

and the numbers of individuals moving between compartments are drawn as
Binomials.  This is the refined scheme of Section 0.2 ("Building the stochastic
model - Refining the model"): the aggregate-count form of the individual-based
Markov chain, identical in distribution to it but costing O(1) per compartment
instead of O(N) per step.  The per-individual loop is *not* used.
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator


def simulate_sir(
    N: int,
    beta: float,
    gamma: float,
    I0: int = 1,
    T: float = 200.0,
    dt: float = 1.0,
    rng: Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate one realization of the stochastic SIR model.

    Returns ``(time, S, I, R)``, each an array of length ``n + 1`` with
    ``n = round(T / dt)``.  The simulation stops evolving once ``I`` hits 0
    (the state is then frozen for the remaining steps).
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(round(T / dt))
    S = np.empty(n + 1)
    I = np.empty(n + 1)
    R = np.empty(n + 1)
    S[0], I[0], R[0] = N - I0, I0, 0
    p_IR = 1.0 - np.exp(-gamma * dt)  # constant: per-infected recovery probability
    for t in range(n):
        if I[t] == 0:  # absorbing: no infectives left, nothing changes
            S[t + 1:] = S[t]
            I[t + 1:] = 0
            R[t + 1:] = R[t]
            break
        p_SI = 1.0 - np.exp(-beta * I[t] / N * dt)  # per-susceptible infection probability
        dSI = rng.binomial(int(S[t]), p_SI)
        dIR = rng.binomial(int(I[t]), p_IR)
        S[t + 1] = S[t] - dSI
        I[t + 1] = I[t] + dSI - dIR
        R[t + 1] = R[t] + dIR
    return np.arange(n + 1) * dt, S, I, R


def final_size(
    N: int,
    beta: float,
    gamma: float,
    I0: int = 1,
    T: float = 200.0,
    dt: float = 1.0,
    rng: Generator | None = None,
) -> int:
    """Total number of individuals ever infected, ``N - S(T)``.

    Light-weight scalar version of :func:`simulate_sir` (no trajectory stored,
    early stop at extinction) for Monte-Carlo studies.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(round(T / dt))
    S, I = N - I0, I0
    p_IR = 1.0 - np.exp(-gamma * dt)
    for _ in range(n):
        if I == 0:
            break
        p_SI = 1.0 - np.exp(-beta * I / N * dt)
        dSI = rng.binomial(int(S), p_SI)
        dIR = rng.binomial(int(I), p_IR)
        S -= dSI
        I += dSI - dIR
    return N - S


def extinction_probability(
    N: int,
    beta: float,
    gamma: float,
    I0: int = 1,
    T: float = 200.0,
    dt: float = 1.0,
    n_reps: int = 2000,
    threshold: float = 0.01,
    rng: Generator | None = None,
) -> tuple[float, float, float]:
    """Estimate P(the disease disappears) by Monte Carlo.

    A run is counted as a *minor outbreak* (the disease disappears) when the
    final size stays below ``threshold * N``.  Returns ``(p_hat, lo, hi)`` with
    a 95% Wald confidence interval on the proportion.
    """
    if rng is None:
        rng = np.random.default_rng()
    minor = 0
    for _ in range(n_reps):
        if final_size(N, beta, gamma, I0, T, dt, rng) < threshold * N:
            minor += 1
    p = minor / n_reps
    half = 1.96 * np.sqrt(p * (1.0 - p) / n_reps)
    return p, max(0.0, p - half), min(1.0, p + half)


def simulate_sirs(
    N: int,
    beta: float,
    gamma: float,
    omega: float,
    I0: int = 1,
    T: float = 200.0,
    dt: float = 1.0,
    rng: Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SIRS model: the SIR model plus waning immunity (R -> S at rate ``omega``).

    This single extra transition relative to :func:`simulate_sir` returns
    recovered individuals to the susceptible pool, which is what lets the disease
    persist and oscillate instead of burning out after one wave (Part I, question
    (b)).  Same Binomial sampling of compartment counts as :func:`simulate_sir`;
    ``omega = 0`` recovers the closed SIR model.  Returns ``(time, S, I, R)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(round(T / dt))
    S = np.empty(n + 1); I = np.empty(n + 1); R = np.empty(n + 1)
    S[0], I[0], R[0] = N - I0, I0, 0
    p_IR = 1.0 - np.exp(-gamma * dt)
    p_RS = 1.0 - np.exp(-omega * dt)
    for t in range(n):
        p_SI = 1.0 - np.exp(-beta * I[t] / N * dt)
        dSI = rng.binomial(int(S[t]), p_SI)
        dIR = rng.binomial(int(I[t]), p_IR)
        dRS = rng.binomial(int(R[t]), p_RS)
        S[t + 1] = S[t] - dSI + dRS
        I[t + 1] = I[t] + dSI - dIR
        R[t + 1] = R[t] + dIR - dRS
    return np.arange(n + 1) * dt, S, I, R


def simulate_sird(
    N: int,
    beta: float,
    gamma: float,
    delta: float,
    I0: int = 1,
    T: float = 200.0,
    dt: float = 1.0,
    rng: Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SIRD model: the SIR model plus disease mortality (I -> D at rate ``delta``).

    Infected individuals leave the I compartment at rate ``gamma + delta``; on
    leaving they recover with probability gamma/(gamma+delta) or die with
    probability delta/(gamma+delta).  This single extra transition (I -> D)
    relative to :func:`simulate_sir` is what lets the disease kill its hosts;
    ``delta = 0`` recovers the closed SIR model.  Same Binomial sampling of
    compartment counts.  Returns ``(time, S, I, R, D)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(round(T / dt))
    S = np.empty(n + 1); I = np.empty(n + 1); R = np.empty(n + 1); D = np.empty(n + 1)
    S[0], I[0], R[0], D[0] = N - I0, I0, 0, 0
    rate_out = gamma + delta
    p_leave = 1.0 - np.exp(-rate_out * dt)
    death_frac = delta / rate_out if rate_out > 0 else 0.0
    for t in range(n):
        if I[t] == 0:                      # absorbing: no infectives, nothing changes
            S[t + 1:] = S[t]; I[t + 1:] = 0; R[t + 1:] = R[t]; D[t + 1:] = D[t]
            break
        p_SI = 1.0 - np.exp(-beta * I[t] / N * dt)
        dSI = rng.binomial(int(S[t]), p_SI)
        d_leave = rng.binomial(int(I[t]), p_leave)     # infectives leaving I this step
        dID = rng.binomial(int(d_leave), death_frac)   # of those, how many die
        dIR = d_leave - dID                            # the rest recover
        S[t + 1] = S[t] - dSI
        I[t + 1] = I[t] + dSI - d_leave
        R[t + 1] = R[t] + dIR
        D[t + 1] = D[t] + dID
    return np.arange(n + 1) * dt, S, I, R, D


def simulate_seird(
    N: int,
    beta_I: float,
    beta_E: float,
    a: float,
    gamma_I: float,
    mu_I: float,
    gamma_E: float = 0.0,
    mu_E: float = 0.0,
    I0: int = 1,
    E0: int = 0,
    T: float = 300.0,
    dt: float = 0.5,
    rng: Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SEIRD model: SEIR with an exposed compartment plus disease mortality (Part II, (a)).

    Susceptibles become *exposed* through contact with infectious **and** exposed
    individuals (force of infection ``(beta_I*I + beta_E*E)/N``).  Exposed individuals
    leave ``E`` at total rate ``Lambda_E = a + gamma_E + mu_E`` -- becoming infectious
    (``a``), recovering (``gamma_E``) or dying (``mu_E``).  Infectious individuals leave
    ``I`` at total rate ``Lambda_I = gamma_I + mu_I`` -- recovering or dying.

    Same refined chain-binomial scheme as :func:`simulate_sir`, but compartments with
    several exits (``E`` and ``I``) are sampled from a **Multinomial** of the current
    count: the probability of *some* transition in ``dt`` is ``1 - exp(-Lambda*dt)`` and
    it is split among the destinations proportionally to their rate.  Setting
    ``beta_E = gamma_E = mu_E = 0`` reduces this to a standard SEIRD parametrization
    (no presymptomatic transmission; exposed only progress to infectious).
    Returns ``(time, S, E, I, R, D)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(round(T / dt))
    S = np.empty(n + 1); E = np.empty(n + 1); I = np.empty(n + 1)
    R = np.empty(n + 1); D = np.empty(n + 1)
    S[0], E[0], I[0], R[0], D[0] = N - I0 - E0, E0, I0, 0, 0

    Lam_E = a + gamma_E + mu_E
    Lam_I = gamma_I + mu_I
    # E -> (I, R, D, stay): split 1 - exp(-Lam_E dt) by the rates a : gamma_E : mu_E
    if Lam_E > 0:
        leaveE = 1.0 - np.exp(-Lam_E * dt)
        pE = [a / Lam_E * leaveE, gamma_E / Lam_E * leaveE, mu_E / Lam_E * leaveE]
        pE.append(1.0 - sum(pE))
    else:
        pE = [0.0, 0.0, 0.0, 1.0]
    # I -> (R, D, stay): split 1 - exp(-Lam_I dt) by the rates gamma_I : mu_I
    if Lam_I > 0:
        leaveI = 1.0 - np.exp(-Lam_I * dt)
        pI = [gamma_I / Lam_I * leaveI, mu_I / Lam_I * leaveI]
        pI.append(1.0 - sum(pI))
    else:
        pI = [0.0, 0.0, 1.0]

    for t in range(n):
        if E[t] == 0 and I[t] == 0:        # absorbing: no one left to infect or progress
            S[t + 1:] = S[t]; E[t + 1:] = 0; I[t + 1:] = 0
            R[t + 1:] = R[t]; D[t + 1:] = D[t]
            break
        p_SE = 1.0 - np.exp(-(beta_I * I[t] + beta_E * E[t]) / N * dt)
        dSE = rng.binomial(int(S[t]), p_SE)
        dEI, dER, dED, _ = rng.multinomial(int(E[t]), pE)
        dIR, dID, _ = rng.multinomial(int(I[t]), pI)
        S[t + 1] = S[t] - dSE
        E[t + 1] = E[t] + dSE - dEI - dER - dED
        I[t + 1] = I[t] + dEI - dIR - dID
        R[t + 1] = R[t] + dER + dIR
        D[t + 1] = D[t] + dED + dID
    return np.arange(n + 1) * dt, S, E, I, R, D


def simulate_sirvd(
    N: int,
    beta: float,
    gamma: float,
    mu: float,
    v,
    I0: int = 1,
    T: float = 300.0,
    dt: float = 0.5,
    rng: Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SIRVD model: the SIRD model plus a vaccinated compartment ``V`` (Part II, (d)).

    Susceptibles now leave ``S`` two ways -- infection (rate ``beta*I/N``) or vaccination
    (rate ``v``) -- so ``S`` has two exits and is sampled from a **Multinomial**:
    ``Lambda_S = beta*I/N + v`` and ``1 - exp(-Lambda_S dt)`` is split between ``I`` and
    ``V`` proportionally to their rates.  Vaccinated individuals are permanently protected
    (``V`` is absorbing).  Infectious individuals recover/die exactly as in
    :func:`simulate_sird`.  The vaccination rate ``v`` may be a constant or a callable
    ``v(t)`` (the campaign effectiveness can change over the epidemic); ``v = 0`` recovers
    the closed SIRD model.  Returns ``(time, S, I, R, V, D)``.
    """
    if rng is None:
        rng = np.random.default_rng()
    v_of = v if callable(v) else (lambda t: v)
    n = int(round(T / dt))
    S = np.empty(n + 1); I = np.empty(n + 1); R = np.empty(n + 1)
    V = np.empty(n + 1); D = np.empty(n + 1)
    S[0], I[0], R[0], V[0], D[0] = N - I0, I0, 0, 0, 0

    Lam_I = gamma + mu                                 # constant: I leaves at this rate
    if Lam_I > 0:
        leaveI = 1.0 - np.exp(-Lam_I * dt)
        pI = [gamma / Lam_I * leaveI, mu / Lam_I * leaveI]
        pI.append(1.0 - sum(pI))
    else:
        pI = [0.0, 0.0, 1.0]

    for t in range(n):
        vt = v_of(t * dt)
        if I[t] == 0 and vt == 0:          # absorbing: no infectives and no vaccination
            S[t + 1:] = S[t]; I[t + 1:] = 0; R[t + 1:] = R[t]
            V[t + 1:] = V[t]; D[t + 1:] = D[t]
            break
        lam_SI = beta * I[t] / N
        Lam_S = lam_SI + vt                            # S leaves towards I or V
        if Lam_S > 0:
            leaveS = 1.0 - np.exp(-Lam_S * dt)
            pS = [lam_SI / Lam_S * leaveS, vt / Lam_S * leaveS]
            pS.append(1.0 - sum(pS))
        else:
            pS = [0.0, 0.0, 1.0]
        dSI, dSV, _ = rng.multinomial(int(S[t]), pS)
        dIR, dID, _ = rng.multinomial(int(I[t]), pI)
        S[t + 1] = S[t] - dSI - dSV
        I[t + 1] = I[t] + dSI - dIR - dID
        R[t + 1] = R[t] + dIR
        V[t + 1] = V[t] + dSV
        D[t + 1] = D[t] + dID
    return np.arange(n + 1) * dt, S, I, R, V, D


# AI Declaration: This module's code was refactored and commented with AI assistance.
# All results and conclusions were independently verified by a human supervisor.
