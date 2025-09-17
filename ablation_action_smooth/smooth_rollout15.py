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
        # 第1组
        0.2,0.2,0,0.06666,0.06666,0.2,0.1333,0.1333,0.06666,0.1333,0.2,0,
        # 第2组
        0.26666,0.26666,0.26666,0.2,0.06666,0.2,0.06666,0.4,0.06666,0.4,0.4,0.133333,
        # 第3组
        0.2,0.2,0.2667,0.2,0.1333,0.2667,0.2,0.2,0.2,0.2,0.1333,0.1333,
        # 第4组
        0.2,0.2,0.4,0.2667,0.2,0.2,0.0667,0.13333,0.4,0.13333,0.0667,0.2,
        # 第5组
        0.2,0.26667,0.46667,0.33333,0.13333,0.2,0.33333,0.2,0.33333,0.33333,0.4,0.33333
    ])

    ours_smooth_norm = np.array([
        # 第1组
        0.8072872129,0.8041315908,0.7329866280,0.7037075250,0.6173614459,0.7254179079,
        0.6915477894,0.7686707502,0.6057066148,0.7936932516,0.7218038512,0.7754772256,
        # 第2组
        0.7867711577,0.7669113702,0.8128852183,0.7751381667,0.8321827221,0.7994417264,
        0.9411857164,0.8405707428,0.9201983341,0.9258231508,0.9833435673,0.8942535994,
        # 第3组
        0.7789914731,0.7777807317,0.7563251064,0.7232686005,0.7742764209,0.8523861393,
        0.8167716761,0.9630232963,0.8111855366,0.7743668973,0.8595016092,0.8009395186,
        # 第4组
        0.7244013626,0.7026923283,0.6797089522,0.6857583652,0.6987361057,0.6423915661,
        0.7149945203,0.5433714342,0.6826619724,0.6235045503,0.6256780,0.6241396381,
        # 第5组
        0.7064137801,0.5483648666,0.7665672596,0.7287030861,0.6419646432,0.6730585167,
        0.6450081893,0.6498003574,0.6171704353,0.6413992784,0.7357329214,0.76804813
    ])

    ours_grad_penalty = np.array([
        # 第1组
        0.0371367891,0.0391762093,0.0322936909,0.0333228864,0.0236790306,0.0295130497,
        0.0294832357,0.0351955012,0.0241375973,0.0391592728,0.0300891113,0.0344966329,
        # 第2组
        0.0370073820,0.0327411451,0.0378783377,0.0325599470,0.0374577171,0.0349881022,
        0.0464144255,0.0377863603,0.0417342939,0.0416896700,0.0485467409,0.0432654933,
        # 第3组
        0.0388617265,0.0339775085,0.0333462162,0.0298638595,0.0352368035,0.0396520715,
        0.0404917064,0.0491561604,0.0365581512,0.0353027133,0.0392326556,0.0362039108,
        # 第4组
        0.0313342245,0.0286231747,0.0276709105,0.0289665414,0.0306873417,0.0241858499,
        0.0329055447,0.0199544695,0.0258961873,0.0271024032,0.0232849121,0.0243947347,
        # 第5组
        0.0284372234,0.0197531593,0.0296345060,0.0313738343,0.0250592256,0.0280267911,
        0.0224728733,0.0259252991,0.0260581433,0.0212824561,0.0311536865,0.0326729676
    ])

    # ---------------- ript ----------------
    ript_success_rate = np.array([
        # 第1组
        0.2666,0.2666,0.06666,0.06666,0.06666,0.06666,0.06666,0.1333,0.2,0.06666,0.1333,0.1333,
        # 第2组
        0.2,0.26666,0.133333,0.133333,0.133333,0.2,0.06666,0.133333,0.06666,0,0.2,0.2,
        # 第3组
        0.1333,0.2,0.3333,0,0.2,0.2667,0.2667,0.1333,0.2,0.1333,0.1333,0.0667,
        # 第4组
        0.3333,0.0667,0.13333,0.13333,0.2,0.2667,0.2,0.4,0.06667,0.3333,
        # 第5组
        0.2,0.2,0.26667,0.26667,0.13333,0.13333,0.4,0.46667,0.26667,0.13333,0.26667,0.2
    ])

    ript_smooth_norm = np.array([
        # 第1组
        0.8073360311,0.5968084636,0.6968815784,0.7108236792,0.7771300097,0.6813353838,
        0.7489831070,0.7322332609,0.5847456766,0.6039304297,0.71593329395,0.68402718936,
        # 第2组
        0.7760546555,0.6670703859,0.7866666719,0.7317571676,0.69116277,0.7681275680,
        0.7695679235,0.8182563936,0.7437956752,0.7285799064,0.7610774715,0.8179467758,
        # 第3组
        0.4113072920,0.5268856273,0.4141778745,0.4762231039,0.4603167521,0.4721409292,
        0.4604266322,0.4559710708,0.4398170853,0.4768950043,0.4421372525,0.4551175582,
        # 第4组
        0.7157744806,0.7507869713,0.7261411159,0.6166020725,0.6689576865,0.6489267103,
        0.6482703822,0.5769937293,0.6448765876,0.7631138988,
        # 第5组
        0.7064253063,0.6105389551,0.7786416524,0.6953730,0.7258917501,0.5888678640,
        0.6410464430,0.6515590769,0.5917748530,0.6610749488,0.5776548054,0.5906446933
    ])

    ript_grad_penalty = np.array([
        # 第1组
        0.0371256166,0.0244530691,0.0315977185,0.0305696559,0.0352156521,0.0313653232,
        0.0305080793,0.0323434100,0.0244131377,0.0255095100,0.0293876085,0.0294417564,
        # 第2组
        0.0381541001,0.0277657022,0.0395310051,0.0321989561,0.0291013216,0.0359870195,
        0.0360568960,0.0371686032,0.0352882909,0.0341930183,0.0353086371,0.0391577670,
        # 第3组
        0.0169038950,0.0192595865,0.0147981343,0.0185267256,0.0160610540,0.0183825325,
        0.0145820692,0.0167577407,0.0167521601,0.0201940183,0.0153993590,0.0182926412,
        # 第4组
        0.0314783799,0.0329927143,0.0299956673,0.0244611049,0.0278127713,0.0254955223,
        0.0246544340,0.0236346223,0.0266045464,0.0331424914,
        # 第5组
        0.0283334856,0.0221480804,0.0286709938,0.0271951904,0.0315843761,0.0204558105,
        0.0242332574,0.0214475904,0.0233025064,0.0253188318,0.0199405731,0.0195156766
    ])  



    results_path = "C:\\Users\\23509\\Desktop\\MiLAB\\iclr2026\\shift_jaco\\smooth_Loss_curve\\"

    stats = plot_jacobian_vs_success_with_ci(ript_smooth_norm, ript_success_rate, 
                                             ours_smooth_norm, ours_success_rate,
                                             save_path=results_path+"smooth_rollout15.png"
                                             )
    print("Regression stats with 95% CI:", stats)
