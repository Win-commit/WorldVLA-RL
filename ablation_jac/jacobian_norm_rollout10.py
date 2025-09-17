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
    plt.xlabel("Jacobian Norm")
    plt.ylabel("Success Rate")
    plt.title("Jacobian Norm vs Task Success Rate with 95% CI")
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
# 实际实验数据
# -----------------
if __name__ == "__main__":





    # ---------------- ours rollout10 ----------------
    ours_success_rate_10 = np.array([
        # 第1组
        0.2,0.2,0.1,0.1,0.0,0.2,0.2,0.3,0.2,0.1,0.0,0.1,
        # 第2组
        0.2,0.1,0.4,0.2,0.1,0.0,0.2,0.1,0.3,0.1,0.1,0.2,
        # 第3组
        0.3,0.2,0.1,0.2,0.2,0.2,0.3,0.3,0.1,0.2,0.1,0.1,
        # 第4组
        0.0,0.2,0.2,0.1,0.3,0.3,0.2,0.4,0.5,0.2,0.2,0.3,
        # 第5组
        0.4,0.2,0.1,0.3,0.2,0.2,0.5,0.2,0.0,0.1,0.3,0.2
    ])


    ours_grad_penalty_10 = np.array([
        # 第1组
        0.03714278916181144,0.03698720088621386,0.03359764500668174,0.02705686421506734,
        0.028716835510048402,0.03432889592727559,0.026237287019428453,0.02895857158460115,
        0.028172643561112255,0.03542137145996094,0.03353912252128322,0.03408361736096834,
        # 第2组
        0.03692406102230674,0.03596737510279605,0.03874427393863076,0.0392940922787315,
        0.039388757002981084,0.04516206370840827,0.039501644644065886,0.04230147913882607,
        0.051394914325914885,0.038560064215409126,0.03085126374897204,0.04434595609966077,
        # 第3组
        0.046249991969058386,0.022598969308953536,0.03721247221294202,0.03533594231856497,
        0.03504180908203125,0.04353181939376028,0.059993743896484375,0.049901560733192844,
        0.055711445055509866,0.04556675961143092,0.039339768259148845,0.03753858996975806,
        # 第4组
        0.03126626265676398,0.03263133355426444,0.03255072699652778,0.03157505236173931,
        0.034590657552083334,0.02977070766212666,0.030427106722133366,0.0319969482421875,
        0.027705485026041667,0.02789668032997533,0.0186911427502065,0.02660952590581939,
        # 第5组
        0.02840275387112185,0.026660986113966556,0.034738239489103616,0.03258856879340278,
        0.028311530463129498,0.026911685341282895,0.026636123657226562,0.025252956814236113,
        0.030696554744944853,0.025668419396924408,0.031867816293839925,0.022535744126523014
    ])

    # ---------------- ript rollout10 ----------------
    ript_success_rate_10 = np.array([
        0.1,0.2,0.1,0.2,0.0,0.3,0.1,0.2,0.1,0.0,0.1,0.2, # 第1组
        0.2,0.0,0.2,0.1,0.2,0.2,0.2,0.2,0.1,0.0,0.0,0.1, # 第2组
        0.1,0.0,0.2,0.2,0.2,0.2,0.3,0.0,0.2,0.1,0.1,0.3, # 第3组
        0.2,0.2,0.0,0.1,0.2,0.3,0.2,0.3,0.2,0.6,0.1,0.3, # 第4组
        0.0,0.3,0.4,0.3,0.4,0.2,0.3,0.5,0.3,0.5,0.3,0.3  # 第5组
    ])


    ript_grad_penalty_10 = np.array([
        0.03385727587728405,0.029019263482862902,0.03914694590111301,0.03541001624103943,
        0.03182246060979446,0.04109102753839132,0.03961014573591469,0.0362780304993091,
        0.029050254147802563,0.033928926112288135,0.023231479388664975,0.02774319583422517,
        0.03815410011693051,0.02884103201486014,0.03557295548288446,0.02973612981582807,
        0.03549183397737455,0.03278876900242554,0.04171190763774671,0.03957648644080529,
        0.035121372767857144,0.03422573732218862,0.033929058885949805,0.027992806783536585,
        0.01690389502862966,0.02361717224121094,0.023618539174397785,0.016581441019917584,
        0.015196675001972854,0.016442982573487443,0.014999629315281414,0.015819170321637425,
        0.017096422389595807,0.017115708423836883,0.021155875898586026,0.013227693327180632,
        0.03147837990208676,0.03018603515625,0.032709523251182156,0.027827930073493084,
        0.030743326459612166,0.028103276302939968,0.027035787225313926,0.02930823356046805,
        0.026395195408871298,0.030273328117999553,0.022280040540193256,0.02713134765625,
        0.02733702952250677,0.030073565456707016,0.02631227773577509,0.025014985867632115,
        0.03156758040832958,0.02552672041223404,0.02991775917795907,0.022067982663390457,
        0.025172529549434268,0.030285373263888887,0.027532958984375,0.0271060624311674
    ])


    results_path = "C:\\Users\\23509\\Desktop\\MiLAB\\iclr2026\\shift_jaco\\jaco_loss_curve\\"

    stats = plot_jacobian_vs_success_with_ci(ript_grad_penalty_10, ript_success_rate_10, 
                                             ours_grad_penalty_10, ours_success_rate_10,
                                             save_path=results_path+"jacobian_success_rollout10_12345.png"
                                             )
    print("Regression stats with 95% CI:", stats)
