"""
Author: Xiyuan Yang   xiyuan_yang@outlook.com
Date: 2025-04-14 19:23:55
LastEditors: Xiyuan Yang   xiyuan_yang@outlook.com
LastEditTime: 2025-04-14 19:24:01
FilePath: /CNN-tutorial/src/convolution.py
Description:
Do you code and make progress today?
Copyright (c) 2025 by Xiyuan Yang, All Rights Reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from scipy.integrate import quad

mlp.use("Agg")


# Original Function f(t)
def original_signal(t):
    return np.sin(2 * np.pi * t)


def echo_kernel2(t, alpha):
    if t >= 0:
        return np.exp(-alpha * t)
    else:
        return 0.0


# Normalize the kernel to ensure its integral is 1
def normalized_echo_kernel(t, alpha):
    # Compute the normalization factor (integral of the kernel from 0 to infinity)
    normalization_factor, _ = quad(lambda tau: echo_kernel2(tau, alpha), 0, np.inf)
    # Return the normalized kernel value
    return echo_kernel2(t, alpha) / normalization_factor


# Convolution with normalized kernel
def convolution(f, g, t_values, alpha):
    h = []
    for t in t_values:
        integral, _ = quad(lambda tau: f(tau) * g(t - tau, alpha), 0, t)
        h.append(integral)
    return np.array(h)


# Plot figure for two pictures
def plotfig(t, f, h, alpha):
    plt.figure(figsize=(10, 6))
    
    # Original Signal
    plt.subplot(2, 1, 1)
    plt.plot(t, f, label="Original Signal", color="blue")
    plt.title("Original Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()

    # Signal with Continuous Echo Effect
    plt.subplot(2, 1, 2)
    plt.plot(t, h, label="Signal with Continuous Echo", color="red")
    plt.title("Signal with Continuous Echo Effect")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"img/Signal_with_Continuous_Effect_alpha={alpha:.2f}.png")
    plt.close()


def main():
    t = np.linspace(0, 5, 500)  # Time range from 0 to 5 seconds
    f = original_signal(t)      # Original signal
    
    # Test different values of alpha
    for i in [0.01, 0.1, 0.3, 0.5, 0.6, 0.8, 1, 1.1, 1.4, 1.5, 2, 2.5, 5]:
        h = convolution(original_signal, normalized_echo_kernel, t, alpha=i)
        plotfig(t, f, h, i)


if __name__ == "__main__":
    main()