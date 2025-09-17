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
        0.0,0.0,0.0,0.33333,0.33333,0.0,0.33333,0.66666,0.0,0.33333,0.66666,0.0,   # 第1组
        0.6667,0.0,0.3333,0.0,0.3333,0.0,0.3333,0.0,0.0,0.3333,0.0,0.0,             # 第2组
        0.6667,0.3333,0.3333,0.0,0.0,0.3333,0.3333,0.0,0.0,0.6667,0.0,0.0,          # 第3组
        0.33333,0.33333,0.33333,0.33333,0.0,0.33333,0.33333,0.0,0.33333,0.0,0.0,0.33333, # 第4组
        0.33333,0.33333,0.33333,0.66666,1.0,0.66666,0.66666,0.66666,0.33333,0.66666,0.66666,0.66666, # 第5组
    ])

    ours_smooth_norm_mean = np.array([
        0.9022366027,0.4244191768,0.9337628644,1.0071430456,0.5815608500,0.6833269393,
        1.0302753790,0.8835662776,0.7133481126,0.8418357730,0.6168817426,0.7197215305,  # 第1组
        0.8668197348,0.7007911145,0.6916181973,0.8210366077,0.8177476790,0.9221994825,
        0.9632072425,0.9581671315,0.9118253112,1.0417738062,0.9960107656,0.9907395915,  # 第2组
        0.7781570654,0.5880224198,0.7107688341,0.7910718531,0.8865117766,0.8118437997,
        0.8310995094,0.7232756402,0.9754097245,1.0976153302,1.1038828460,0.7213502655,  # 第3组
        0.6995919078,0.5451029707,0.9540261412,0.7469428770,0.9273726653,0.7604492756,
        0.5499441543,0.7941496987,0.7565105731,0.6792230513,0.7121520649,0.6288519071,  # 第4组
        0.6150905305,0.6330096910,0.8463638099,0.4700856673,0.8413410790,0.6446714723,
        0.7822698090,0.6804069542,0.5925832553,0.5427562237,0.5153848203,0.9229572100,  # 第5组
    ])

    ours_grad_penalty = np.array([
        0.0496737832,0.0171566087,0.0438565706,0.0489536085,0.0225719653,0.0267255683,
        0.0478319871,0.0349091205,0.0365954951,0.0312337373,0.0268714139,0.0307316027,  # 第1组
        0.0457494635,0.0322912116,0.0389887157,0.0397626977,0.0338142348,0.0518356483,
        0.0532933285,0.0446218189,0.0499119006,0.0465826737,0.0545001783,0.0535309157,  # 第2组
        0.0443619176,0.0221147035,0.0355606079,0.0363473390,0.0400682751,0.0388024983,
        0.0335150267,0.0273062762,0.0540986312,0.0555883709,0.0479449222,0.0270710000,  # 第3组
        0.0301782708,0.0189522191,0.0413921921,0.0306632502,0.0425246389,0.0307000249,
        0.0244961300,0.0384363939,0.0355192486,0.0289639925,0.0315746508,0.0242269607,  # 第4组
        0.0243793287,0.0258131285,0.0357864781,0.0133080524,0.0379875585,0.0251870292,
        0.0401313179,0.0215086179,0.0212264288,0.0210367504,0.0204131298,0.0393792203,  # 第5组
    ])

    # ---------------- ript ----------------
    ript_success_rate = np.array([
        0.0,0.66666,0.33333,0.0,0.33333,0.66666,0.0,0.66666,0.33333,0.66666,0.33333,0.33333,   # 第1组
        0.0,0.0,0.3333,0.0,0.6667,0.0,0.3333,0.0,0.0,0.3333,0.0,0.6667,                     # 第2组
        0.3333,0.3333,0.0,0.0,0.0,0.3333,0.6667,0.0,0.0,0.0,0.0,0.3333,                      # 第3组
        0.3333,0.3333,0.0,0.6667,0.3333,0.0,0.6667,0.3333,0.0,0.3333,0.3333,0.3333,          # 第4组
        0.66666,0.33333,0.33333,0.0,0.66666,0.66666,0.33333,0.66666,0.33333,0.33333,0.33333,0.0, # 第5组
    ])

    ript_smooth_norm_mean = np.array([
        0.5899969093,0.9164260807,0.6667180674,0.4096347248,0.5247076204,0.8817366134,
        0.7709509081,0.7778874167,0.7911206244,1.0225602434,0.7124377659,0.6047750627,  # 第1组
        0.5815413107,0.7546747395,0.6863361957,0.8070509680,0.6960321571,0.5594339745,
        0.9442246189,0.9293005787,0.6791838480,0.5289668512,0.5823966037,0.7315618293,  # 第2组
        0.3563354783,0.5892578530,0.5592771169,0.3719348267,0.4440027378,0.5080383421,
        0.6598042954,0.5802323271,0.5939009377,0.4000365400,0.3683428579,0.3260278021,  # 第3组
        0.6997988873,0.6413773428,0.7432105653,0.6802852557,0.7576223350,0.6352578306,
        0.7451522163,0.6272735304,0.6763891066,0.7996218028,0.8282378997,0.6076046146,  # 第4组
        0.6321568436,1.0863079129,0.5293894255,0.8176733476,0.6961012466,0.5348172505,
        0.7186285301,0.9267442424,0.6831440752,0.6078757814,0.5434600308,0.6848000998,  # 第5组
    ])

    ript_grad_penalty = np.array([
        0.0211305164,0.0477289869,0.0283754446,0.0135150120,0.0202298658,0.0382153279,
        0.0376365890,0.0273558748,0.0388719487,0.0543819118,0.0304029310,0.0246164244,  # 第1组
        0.0269100483,0.0330351142,0.0245174182,0.0374263218,0.0305144746,0.0256603911,
        0.0465922701,0.0423029307,0.0268014669,0.0209800591,0.0266782014,0.0371010,    # 第2组
        0.0109293948,0.0215636279,0.0230957834,0.0136404382,0.0182419742,0.0190642627,
        0.0292713254,0.0234234060,0.0265670321,0.0141342625,0.0121293657,0.0112872877,  # 第3组
        0.0306305132,0.0249228227,0.0288049269,0.0262730247,0.0302041692,0.0222626736,
        0.0312658611,0.0284033324,0.0243926892,0.0409849066,0.0357530493,0.0247058868,  # 第4组
        0.0262733061,0.0584990023,0.0193172988,0.0329035478,0.0288732385,0.0170703130,
        0.0314140815,0.0359946571,0.0222581113,0.0229224128,0.0193714451,0.0246297078,  # 第5组
    ])






    results_path = "C:\\Users\\23509\\Desktop\\MiLAB\\iclr2026\\shift_jaco\\jaco_loss_curve\\"

    stats = plot_jacobian_vs_success_with_ci(ript_grad_penalty, ript_success_rate, 
                                             ours_grad_penalty, ours_success_rate,
                                             save_path=results_path+"jacobian_rollout3.png"
                                             )
    print("Regression stats with 95% CI:", stats)
