import matplotlib.pyplot as plt

plt.style.use('default')

def create_visualization(shear_rate, shear_stress, ty_YPL, K_YPL, n_YPL, K_PL, n_PL, PV, YP):
    shear_stress_calc_YPL = YPLfunction(shear_rate, ty_YPL, K_YPL, n_YPL)
    shear_stress_calc_PL = PLfunction(shear_rate, K_PL, n_PL)
    shear_stress_calc_BP = BPfunction(shear_rate, PV, YP)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(x=shear_rate, y=shear_stress,
               label="Measured viscometer data", color="red")

    ax.plot(shear_rate, shear_stress_calc_YPL,
            label="Yield Power-law model fit", color="blue")

    ax.plot(shear_rate, shear_stress_calc_PL,
            label="Power-law model fit", color="orange")

    ax.plot(shear_rate, shear_stress_calc_BP,
            label="Bingham Plastic model Fit", color="green")

    ax.set_xlabel("Shear Rate (1/s)")
    ax.set_ylabel("Shear Stress (Pa)")
    ax.set_xlim(0, round(max(shear_rate) + 40, 0))
    ax.set_ylim(0, round(max(shear_stress) + 10, 0))
    ax.legend()

    return fig