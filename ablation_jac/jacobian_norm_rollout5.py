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
    plt.xlabel("jacobian Norm")
    plt.ylabel("Success Rate")
    plt.title("jacobian Norm vs Task Success Rate with 95% CI")
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
        0.4,0.4,0.2,0.0,0.4,0.4,0.4,0.6,0.2,0.2,0.0,0.2,   # 第1组
        0.2,0.0,0.4,0.2,0.2,0.0,0.4,0.2,0.2,0.0,0.2,0.2,   # 第2组
        0.2,0.4,0.0,0.0,0.4,0.0,0.2,0.4,0.0,0.2,0.2,0.4,   # 第3组
        0.0,0.0,0.4,0.0,0.0,0.4,0.6,0.2,0.8,0.4,0.4,0.6,   # 第4组
        0.4,0.4,0.4,0.6,0.4,0.4,0.8,0.4,0.2,0.4,0.2,0.4,   # 第5组
    ])

    ours_smooth_norm_mean = np.array([
        0.7998817519,0.9121973628,0.7540710972,0.8622591535,0.6080204502,
        0.7514538798,0.8526982571,0.8800430891,0.7526497910,0.9536804673,
        0.8346476434,0.8088365053,  # 第1组
        0.7774806954,0.6824445743,0.7865053925,0.7325440798,0.9774902085,
        0.7468313188,0.8537806255,0.8448285828,0.8454745379,0.7866179691,
        0.8083597663,0.8662365082,  # 第2组
        0.7677319760,0.7452189220,0.7484385950,0.8761082660,0.7139856000,
        0.9108510740,0.7639523710,1.0726529240,0.7071695810,0.8001064660,
        0.8958093030,0.7795781310,  # 第3组
        0.7301452096,0.7028802262,0.7567646815,0.6922059376,0.7435621860,
        0.6984822533,0.8256308879,0.7468709123,0.7503513711,0.6811007314,
        0.5840351924,0.6650747658,  # 第4组
        0.6926937353,0.7471883255,0.8290212531,0.6764120831,0.7865841590,
        0.7463129184,0.7072681813,0.6172273863,0.5817771200,0.6048050910,
        0.5941401603,0.6411213113,  # 第5组
    ])

    ours_grad_penalty = np.array([
        0.0342360588,0.0358303472,0.0348329042,0.0421425669,0.0226760663,
        0.0345125700,0.0347153513,0.0367952648,0.0286321542,0.0400071395,
        0.0403610029,0.0414673655,  # 第1组
        0.0366244065,0.0300136365,0.0386015240,0.0324893751,0.0494894441,
        0.0354950553,0.0466679021,0.0368155061,0.0444813779,0.0358940928,
        0.0403789721,0.0399834482,  # 第2组
        0.0366227000,0.0343312720,0.0363299720,0.0435756880,0.0298048320,
        0.0413882800,0.0344262370,0.0589502230,0.0340024040,0.0315601050,
        0.0429144970,0.0327406430,  # 第3组
        0.0333457746,0.0294050919,0.0331138162,0.0275543032,0.0312386307,
        0.0288755144,0.0360966485,0.0328812487,0.0289537650,0.0269753356,
        0.0239234760,0.0271865515,  # 第4组
        0.0276376825,0.0302408447,0.0371641862,0.0261043495,0.0283810104,
        0.0251177145,0.0294420343,0.0222981771,0.0208176833,0.0188912405,
        0.0209191939,0.0251246385,  # 第5组
    ])

    # ---------------- ript ----------------
    ript_success_rate = np.array([
        0.2,0.2,0.2,0.2,0.2,0.2,0.0,0.4,0.0,0.4,0.2,0.2,   # 第1组
        0.2,0.2,0.0,0.4,0.0,0.2,0.2,0.2,0.4,0.2,0.2,0.2,   # 第2组
        0.2,0.2,0.2,0.2,0.2,0.4,0.2,0.2,0.0,0.0,0.2,0.0,   # 第3组
        0.0,0.2,0.2,0.4,0.2,0.2,0.0,0.6,0.2,0.0,0.2,0.2,   # 第4组
        0.4,0.4,0.2,0.0,0.4,0.4,0.4,0.6,0.2,0.2,0.0,0.2,   # 第5组
    ])

    ript_smooth_norm_mean = np.array([
        0.7999028332,0.7568748992,0.7458218245,0.9293936986,0.8876231309,
        0.7574876270,0.5848209008,0.6511510182,0.6080919264,0.6379519805,
        0.8139682605,0.6850535606,  # 第1组
        0.7775340157,0.7538599299,0.6559190883,0.6133401698,0.7750671859,
        0.8134011661,0.6944611081,0.8071166080,0.8364943785,0.5724214383,
        0.9278378582,0.8962115471,  # 第2组
        0.4481834864,0.4406598585,0.4332924188,0.4025503869,0.4107858379,
        0.4005283286,0.3827364370,0.5900844580,0.4740172017,0.4113846539,
        0.4794077567,0.5954975441,  # 第3组
        0.7301938417,0.6873948767,0.6808641870,0.5900009254,0.6453989605,
        0.6390067910,0.5970026825,0.6516676821,0.6503182757,0.6581173523,
        0.6286708748,0.7549156214,  # 第4组
        0.4790264356,0.6807045500,0.7531975409,0.8322451702,0.6691151767,
        0.7346273697,0.5769510417,0.7576024488,0.9011616042,0.6733530087,
        0.6218964112,0.5614343580,  # 第5组
    ])

    ript_grad_penalty = np.array([
        0.0343790149,0.0295451482,0.0306601671,0.0545319719,0.0439971924,
        0.0310421898,0.0251287993,0.0233144124,0.0212004152,0.0311646124,
        0.0323265584,0.0271572988,  # 第1组
        0.0368736669,0.0331521370,0.0279069198,0.0241218366,0.0405446103,
        0.0392414896,0.0283934945,0.0394274059,0.0409511767,0.0252426549,
        0.0493100819,0.0448109476,  # 第2组
        0.0201671163,0.0150975971,0.0149596458,0.0170276921,0.0148683436,
        0.0133535896,0.0132317114,0.0244531631,0.0189017657,0.0141187646,
        0.0170180242,0.0248210362,  # 第3组
        0.0335679305,0.0262575651,0.0289853749,0.0214397113,0.0276646821,
        0.0255045640,0.0237590896,0.0266908186,0.0248093741,0.0264447865,
        0.0266658381,0.0362430271,  # 第4组
        0.0170773097,0.0253517151,0.0325629711,0.0351328816,0.0223122056,
        0.0310683379,0.0188632011,0.0298162227,0.0438791179,0.0280347063,
        0.0234720139,0.0213161922,  # 第5组
    ])





    results_path = "C:\\Users\\23509\\Desktop\\MiLAB\\iclr2026\\shift_jaco\\jaco_loss_curve\\"

    stats = plot_jacobian_vs_success_with_ci(ript_grad_penalty, ript_success_rate, 
                                             ours_grad_penalty, ours_success_rate,
                                             save_path=results_path+"jacobian_rollout5.png"
                                             )
    print("Regression stats with 95% CI:", stats)
