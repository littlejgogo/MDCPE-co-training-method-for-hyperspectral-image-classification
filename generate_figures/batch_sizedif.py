import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

T_sa = np.array([0, 3, 5, 7, 9, 11])

power_sa = np.array([92.00262, 89.180327, 90.1639344, 92.459016, 89.836065, 88.85245])
power_pu = np.array([88.65546, 90.33613, 86.134453, 88.445378, 86.97478, 88.8655])
power_pc = np.array([97.7153736, 97.00142789, 97.858162, 97.524988, 96.953831, 97.620180])

axes = plt.gca()
axes.grid(True)
axes.set_xlim([0, 11])
axes.set_ylim([84.5, 99.5])


plt.xlabel("The Number of Updated Samples in Each Class")
plt.ylabel("Overall Accuracy (%)")
plt.plot(T_sa, power_sa, 'go')
plt.plot(T_sa, power_sa, color='green', label='SA')
plt.plot(T_sa, power_pu, 'rD')
plt.plot(T_sa, power_pu, color='red', label='PU')
plt.plot(T_sa, power_pc, 'bs')
plt.plot(T_sa, power_pc, color='blue', label='PC')

plt.annotate("(7, 92.46%)", xy=(7, 92.459016), xytext=(7.5, 91.5), arrowprops=dict(facecolor='green', shrink=0.1))
plt.annotate("(3, 90.34%)", xy=(3, 90.33613), xytext=(3.5, 89.5), arrowprops=dict(facecolor='red', shrink=0.1))
plt.annotate("(5, 97.86%)", xy=(5, 97.858162), xytext=(5.5, 96.8), arrowprops=dict(facecolor='blue', shrink=0.1))
plt.legend(loc='best', ncol=1, fancybox=True, shadow=True, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize='small')
# plt.show()
plt.savefig("upeachclass1.jpg")

