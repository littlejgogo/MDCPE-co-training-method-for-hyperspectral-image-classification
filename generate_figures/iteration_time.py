import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

T_sa = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

mdcpe_sa = np.array([92.00262, 92.131147, 93.44262, 94.098360655, 92.1311475, 90.819672, 91.803278, 92.13175411, 92.45901639,
                     90.49180327, 91.873207])
mdcpe_pu = np.array([88.65546, 89.915966, 89.0756302, 91.1764705, 89.076532, 88.65546218, 90.12605042, 87.3949579,
                     88.02521008, 86.5504621, 90.00336134])
mdcpe_pc = np.array([97.007153, 97.14421789, 97.287006, 98.2865302, 98.429319, 97.810566, 97.57258448, 97.477217039,
                     97.33460257, 96.57258448, 97.001427])
dcpe_sa = np.array([92.00262, 91.14754, 91.8032786, 92.45901639, 93.44262, 92.4931634, 90.4918032, 92.7868852,
                    91.14754098, 90.1639344, 89.795216])
dcpe_pu = np.array([88.65546, 87.394957, 87.15126, 88.235294, 88.655462, 88.8655942, 89.2857142, 89.495798, 88.8655462,
                    86.7647058, 86.13445378])
dcpe_pc = np.array([97.007153, 97.5725844, 96.8110423, 97.3821989, 96.144693, 98.096144, 97.57258448, 97.287006,
                    97.95335554, 96.33460257, 97.000951927])
axes = plt.gca()
axes.grid(True)
axes.set_xlim([-0.5, 10.5])
axes.set_ylim([82, 100])

plt.xlabel("The Iterations of Co-training")
plt.ylabel("Overall Accuracy (%)")
plt.plot(T_sa, mdcpe_sa, 'go')
plt.plot(T_sa, mdcpe_sa, color='green', label='MDCPE on SA', linestyle="-")
plt.plot(T_sa, mdcpe_pu, 'rD')
plt.plot(T_sa, mdcpe_pu, color='red', label='MDCPE on PU', linestyle="-")
plt.plot(T_sa, mdcpe_pc, 'bs')
plt.plot(T_sa, mdcpe_pc, color='blue', label='MDCPE on PC', linestyle="-")

plt.plot(T_sa, dcpe_sa, 'go')
plt.plot(T_sa, dcpe_sa, color='green', label='DCPE on SA', linestyle="--")
plt.plot(T_sa, dcpe_pu, 'rD')
plt.plot(T_sa, dcpe_pu, color='red', label='DCPE on PU', linestyle="--")
plt.plot(T_sa, dcpe_pc, 'bs')
plt.plot(T_sa, dcpe_pc, color='blue', label='DCPE on PC', linestyle="--")


plt.annotate("(3, 94.10%)", xy=(3, 94.098360655), xytext=(3.5, 95), arrowprops=dict(facecolor='green', shrink=0.1))
plt.annotate("(3, 91.18%)", xy=(3, 91.1764705), xytext=(3.5, 90.1), arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate("(4, 98.43%)", xy=(4, 98.429319), xytext=(4.5, 99.4), arrowprops=dict(facecolor='blue', shrink=0.1))
plt.annotate("(4, 93.44%)", xy=(4, 93.44262), xytext=(4.5, 94.4), arrowprops=dict(facecolor='green', shrink=0.1))
plt.annotate("(7, 89.50%)", xy=(7, 89.495798), xytext=(7.5, 88.4), arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate("(5, 98.10%)", xy=(5, 98.096144), xytext=(5.5, 99), arrowprops=dict(facecolor='blue', shrink=0.1))
plt.legend(loc='lower right', ncol=2, fancybox=True, shadow=True, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize='small')
# plt.show()
plt.savefig("iter_acc3.jpg")

