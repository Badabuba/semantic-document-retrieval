import numpy as np
import matplotlib.pyplot as plt


def main(singular_values, filename):
    squared_sv = singular_values ** 2
    total_variance = np.sum(squared_sv)
    cumulative_variance = (np.cumsum(squared_sv) / total_variance) * 100

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_blue = '#1f77b4'
    ax1.set_xlabel('Principal Component Index', fontsize=12, labelpad=10)
    ax1.set_ylabel('Singular Value ($\sigma_i$)', color=color_blue, fontsize=12)
    line1 = ax1.plot(range(1, len(singular_values) + 1), singular_values, 
                    color=color_blue, linewidth=2, marker='o', markersize=4, label='Singular Value')
    ax1.tick_params(axis='y', labelcolor=color_blue)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color_orange = '#ff7f0e'
    ax2.set_ylabel('Cumulative Explained Variance (%)', color=color_orange, fontsize=12)
    line2 = ax2.plot(range(1, len(singular_values) + 1), cumulative_variance, 
                    color=color_orange, linewidth=2, linestyle='--', label='Cumulative Variance')
    ax2.tick_params(axis='y', labelcolor=color_orange)

    plt.title("LSA Model Compression: Information Captured per Dimension ($k$)", fontsize=14, pad=15, weight='bold')
    plt.tight_layout()

    plt.savefig(filename, dpi=300)

if __name__ == "__main__":
    singular_values = np.array([
        2.36072369, 1.54507531, 1.45116258, 1.39648874, 1.34440871, 1.25716563,
        1.20276133, 1.17591227, 1.15202945, 1.138523,   1.12928878, 1.11784908,
        1.07911825, 1.06426173, 1.05168566, 1.01800964, 0.99985488, 0.9857788,
        0.97181213, 0.96807347, 0.95207499, 0.94970026, 0.93303667, 0.92871788,
        0.92386595, 0.91229375, 0.90810692, 0.90281383, 0.89948351, 0.89415589,
        0.88399043, 0.87846519, 0.87215029, 0.87022028, 0.86680391, 0.86614559,
        0.86204215, 0.85846769, 0.85548761, 0.85124318, 0.84783851, 0.84153478,
        0.84021253, 0.83916578, 0.83520924, 0.83359521, 0.83175629, 0.83005744,
        0.82830773, 0.82309451, 0.82033979, 0.81636195, 0.81450798, 0.81367938,
        0.81212643, 0.80873787, 0.80533332, 0.80290838, 0.80039916, 0.79622226,
        0.79431593, 0.79299924, 0.79162005, 0.78929846, 0.7874477,  0.78385187,
        0.78329052, 0.78329369, 0.78151404, 0.78029532, 0.77682638, 0.77376324,
        0.77190498, 0.77093224, 0.76924366, 0.76710125, 0.76666338, 0.76603155,
        0.76209578, 0.76057472, 0.76017918, 0.75730446, 0.75606863, 0.75269035,
        0.75151908, 0.75112072, 0.74776727, 0.74706682, 0.7451871,  0.74269892,
        0.74210381, 0.74101523, 0.73978502, 0.73800396, 0.73794405, 0.73526384,
        0.73411536, 0.73246903, 0.73127995, 0.72930395
    ])

    main(singular_values, "plot.png")

