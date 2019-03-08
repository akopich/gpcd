import matplotlib.pyplot as plt

delta5 = [0.900416092494, 0.393647132356, 0.199288465027, 0.0999538013804]
power5 = [1.0, 0.8437, 0.11, 0.0225]
prec5 = [1.0, 1.1, 75, 293]

delta10 = [0.393647132356, 0.199288465027, 0.0999538013804, 0.0500155837833]
power10 = [0.99875, 0.55, 0.02875, 0.0125]
prec10 = [1.27, 36, 150, 338]

delta20 = delta10
power20 = [1.0, 0.97, 0.19, 0.02]
prec20 = [1.1, 2.4, 29, 292]

delta40 = [0.199288465027, 0.0999538013804, 0.0500155837833]
power40 = [1.0, 0.73, 0.071]
prec40 = [2.5, 22, 188]

plt.interactive(False)

plt.plot(delta5, power5, 'o', linestyle='-')
plt.plot(delta10, power10, 'o', linestyle='-')
plt.plot(delta20, power20, 'o', linestyle='-')
plt.plot(delta40, power40, 'o', linestyle='-')
plt.ylabel("Power")
plt.xlabel("Observable break extent")
plt.xticks([0.9, 0.4, 0.2, 0.1, 0.05])
plt.legend(['n = 5', 'n = 10', 'n = 20', 'n = 40'], loc='lower right')
plt.show()


plt.plot(delta5, prec5, 'o', linestyle='-')
plt.plot(delta10, prec10, 'o', linestyle='-')
plt.plot(delta20, prec20, 'o', linestyle='-')
plt.plot(delta40, prec40, 'o', linestyle='-')
plt.ylabel("RMSE")
plt.xlabel("Observable break extent")
plt.xticks([0.9, 0.4, 0.2, 0.1, 0.05])
plt.legend(['n = 5', 'n = 10', 'n = 20', 'n = 40'], loc='upper right')
plt.show()
