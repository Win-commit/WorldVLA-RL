import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.utils import resample

def plot_jacobian_vs_success_with_ci(jac_base, success_base, jac_ours, success_ours, save_path="jacobian_success_ci.pdf"):
    """
    Scatter + Regression + 95% CI for Baseline vs Ours
    """
    def compute_ci(x, y, n_boot=1000, alpha=0.05):
        """
        Bootstrap 95% CI for linear regression line
        """
        x = np.array(x)
        y = np.array(y)
        lines = []
        N = len(x)
        for _ in range(n_boot):
            idx = np.random.choice(N, N, replace=True)
            slope, intercept, _, _, _ = linregress(x[idx], y[idx])
            lines.append((slope, intercept))
        lines = np.array(lines)
        # Compute CI for predicted y at each x
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_preds = np.array([slope * x_fit + intercept for slope, intercept in lines])
        lower = np.percentile(y_preds, 100*alpha/2, axis=0)
        upper = np.percentile(y_preds, 100*(1-alpha/2), axis=0)
        # Also return mean regression line
        mean_slope = np.mean(lines[:,0])
        mean_intercept = np.mean(lines[:,1])
        y_mean = mean_slope * x_fit + mean_intercept
        return x_fit, y_mean, lower, upper, mean_slope, mean_intercept

    # Baseline
    x_base, y_base_mean, y_base_lower, y_base_upper, slope_base, intercept_base = compute_ci(jac_base, success_base)
    r_base, _, _, _ = linregress(jac_base, success_base)[:4]

    # Ours
    x_ours, y_ours_mean, y_ours_lower, y_ours_upper, slope_ours, intercept_ours = compute_ci(jac_ours, success_ours)
    r_ours, _, _, _ = linregress(jac_ours, success_ours)[:4]

    # Plot
    plt.figure(figsize=(7,5))
    # Scatter
    plt.scatter(jac_base, success_base, c='steelblue', s=30, alpha=0.7, label="Baseline")
    plt.scatter(jac_ours, success_ours, c='darkorange', s=30, alpha=0.7, label="Ours")
    # Regression + CI
    plt.plot(x_base, y_base_mean, 'b--', lw=2)
    plt.fill_between(x_base, y_base_lower, y_base_upper, color='blue', alpha=0.2)
    plt.plot(x_ours, y_ours_mean, 'r--', lw=2)
    plt.fill_between(x_ours, y_ours_lower, y_ours_upper, color='orange', alpha=0.2)
    
    # Labels
    plt.xlabel("Smooth Norm")
    plt.ylabel("Success Rate")
    plt.title("Smooth Norm vs Task Success Rate with 95% CI")
    plt.legend()

    # Stats text
    plt.text(0.02, 0.95,
             f"Baseline: slope={slope_base:.3f}, R~{r_base:.2f}\nOurs: slope={slope_ours:.3f}, R~{r_ours:.2f}",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {
        "baseline": {"slope": slope_base, "intercept": intercept_base},
        "ours": {"slope": slope_ours, "intercept": intercept_ours}
    }


# -----------------
# 示例数据
# -----------------
if __name__ == "__main__":



# ---------------- ours ----------------
    ours_success_rate = np.array([
        0.2,0.2,0.1,0.1,0.0,0.2,0.2,0.3,0.2,0.1,0.0,0.1,   # 第1组
        0.2,0.1,0.4,0.2,0.1,0.0,0.2,0.1,0.3,0.1,0.1,0.2,   # 第2组
        0.3,0.2,0.1,0.2,0.2,0.2,0.3,0.3,0.1,0.2,0.1,0.1,   # 第3组
        0.0,0.2,0.1,0.3,0.3,0.3,0.2,0.4,0.5,0.2,0.2,0.3,   # 第4组
        0.4,0.2,0.1,0.3,0.2,0.2,0.5,0.2,0.0,0.1,0.3,0.2,   # 第5组
    ])

    ours_smooth_norm_mean = np.array([
        0.8073240285,0.8051936718,0.8064017675,0.6762253449,0.6988826,
        0.7820363656,0.6753254351,0.7270665221,0.6937722373,0.7469171522,
        0.7932041556,0.7589985324,  # 第1组
        0.78675756,0.761477478,0.8331739538,0.8224933843,0.8760455215,
        0.9256227269,0.8391239077,0.881308321,1.0302419793,0.8433960461,
        0.7505313569,0.8872394236,  # 第2组
        0.8466270404,0.5899807801,0.7480129115,0.7949777808,0.7808290357,
        0.9287435425,0.9671710298,0.9675051426,1.2122042153,0.9100667971,
        0.9985995654,0.9083706103,  # 第3组
        0.7158198048,0.7608841532,0.7428373818,0.7440467,0.7340776721,
        0.6533538503,0.7031703708,0.7332332228,0.6314281831,0.6442923165,
        0.5458938197,0.6498027806,  # 第4组
        0.7065021135,0.725286833,0.757087997,0.7372633698,0.7135374904,
        0.6916872183,0.6653893067,0.6712933885,0.7054139123,0.6744911128,
        0.7172144865,0.5953618655,  # 第5组
    ])

    ours_grad_penalty = np.array([
        0.0371427892,0.0369872009,0.0335976450,0.0270568642,0.0287168355,
        0.0343288959,0.0262372870,0.0289585716,0.0281726436,0.0354213715,
        0.0335391225,0.0340836174,  # 第1组
        0.0369240610,0.0359673751,0.0387442739,0.0392940923,0.0393887570,
        0.0451620637,0.0395016446,0.0423014791,0.0513949143,0.0385600642,
        0.0308512637,0.0443459561,  # 第2组
        0.0462499920,0.0225989693,0.0372124722,0.0353359423,0.0350418091,
        0.0435318194,0.0599937439,0.0499015607,0.0557114451,0.0455667596,
        0.0393397683,0.03753859,    # 第3组
        0.0312662627,0.0326313336,0.0325507270,0.0315750524,0.0345906576,
        0.0297707077,0.0304271067,0.0319969482,0.0277054850,0.0278966803,
        0.0186911428,0.0266095259,  # 第4组
        0.0284027539,0.0266609861,0.0347382395,0.0325885688,0.0283115305,
        0.0269116853,0.0266361237,0.0252529568,0.0306965547,0.0256684194,
        0.0318678163,0.0225357441,  # 第5组
    ])

    # ---------------- ript ----------------
    ript_success_rate = np.array([
        0.1,0.2,0.1,0.2,0.0,0.3,0.1,0.2,0.1,0.0,0.1,0.2,  # 第1组
        0.2,0.0,0.2,0.1,0.2,0.2,0.2,0.2,0.1,0.0,0.0,0.1,  # 第2组
        0.1,0.0,0.2,0.2,0.2,0.2,0.3,0.0,0.2,0.1,0.1,0.3,  # 第3组
        0.2,0.2,0.0,0.1,0.2,0.3,0.2,0.3,0.2,0.6,0.1,0.3,  # 第4组
        0.0,0.3,0.4,0.3,0.4,0.2,0.3,0.5,0.3,0.5,0.3,0.3,  # 第5组
    ])

    ript_smooth_norm_mean = np.array([
        0.7635573,0.64563294,0.82525084,0.78183030,0.73005880,
        0.84552081,0.78756676,0.81336751,0.68484585,0.71339136,
        0.61994259,0.71148286,  # 第1组
        0.77605466,0.69449476,0.78711071,0.67274655,0.78670098,
        0.75032174,0.83508571,0.85096528,0.75879952,0.76875946,
        0.72719137,0.66377673,  # 第2组
        0.41130729,0.55573465,0.54219091,0.42272118,0.41648613,
        0.45657003,0.39533913,0.40172641,0.38923649,0.44184814,
        0.48588453,0.37658575,  # 第3组
        0.71577448,0.69799753,0.74723041,0.68276853,0.72102984,
        0.70093849,0.60757792,0.68922566,0.67263536,0.67165134,
        0.58282758,0.58204691,  # 第4组
        0.69826885,0.75655734,0.68470636,0.65608342,0.76565440,
        0.62812736,0.68624970,0.64303558,0.65773888,0.73491915,
        0.66291032,0.68260613,  # 第5组
    ])

    ript_grad_penalty = np.array([
        0.03385728,0.02901926,0.03914695,0.03541002,0.03182246,
        0.04109103,0.03961015,0.03627803,0.02905025,0.03392893,
        0.02323148,0.02774320,  # 第1组
        0.03815410,0.02884103,0.03557296,0.02973613,0.03549183,
        0.03278877,0.04171191,0.03957649,0.03512137,0.03422574,
        0.03392906,0.02799281,  # 第2组
        0.01690390,0.02361717,0.02361854,0.01658144,0.01519668,
        0.01644298,0.01499963,0.01581917,0.01709642,0.01711571,
        0.02115588,0.01322769,  # 第3组
        0.03147838,0.03018604,0.03270952,0.02782793,0.03074333,
        0.02810328,0.02703579,0.02930823,0.02639520,0.03027333,
        0.02228004,0.02713135,  # 第4组
        0.02733703,0.03007357,0.02631228,0.02501499,0.03156758,
        0.02552672,0.02991776,0.02206798,0.02517253,0.03028537,
        0.02753296,0.02710606,  # 第5组
    ])  



    results_path = "C:\\Users\\23509\\Desktop\\MiLAB\\iclr2026\\shift_jaco\\smooth_Loss_curve\\"

    stats = plot_jacobian_vs_success_with_ci(ript_smooth_norm_mean, ript_success_rate, 
                                             ours_smooth_norm_mean, ours_success_rate,
                                             save_path=results_path+"smooth_rollout10.png"
                                             )
    print("Regression stats with 95% CI:", stats)
