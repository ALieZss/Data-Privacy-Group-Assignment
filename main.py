import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bisect

# SumDP
def sumdp_sum(data, epsilon, U, beta=0.1):
    segments = []
    start = 1
    while start <= U:
        end = min(2 * start - 1, U)
        segments.append((start, end))
        start = end + 1


    sorted_data = np.sort(data)
    prefix_sum = np.concatenate([[0], np.cumsum(sorted_data)])

    seg_sums = []
    for (low, high) in segments:
        lo_idx = bisect.bisect_left(sorted_data, low)
        hi_idx = bisect.bisect_right(sorted_data, high)
        if lo_idx == hi_idx:
            seg_sums.append(0.0)
        else:
            seg_sum = prefix_sum[hi_idx] - prefix_sum[lo_idx]
            seg_sums.append(seg_sum)
    seg_sums = np.array(seg_sums)


    noisy_sums = np.empty(len(segments))
    for i, (low, high) in enumerate(segments):
        scale = high / epsilon
        noise = np.random.laplace(0, scale)
        noisy_sums[i] = seg_sums[i] + noise


    m = len(segments)
    thresholds = [1.3 * (high / epsilon) * np.log((2 * m + 1) / beta) for (_, high) in segments]
    chosen_idx = -1
    for j in range(m - 1, -1, -1):
        if noisy_sums[j] > thresholds[j]:
            chosen_idx = j
            break

    result = noisy_sums[:chosen_idx + 1].sum() if chosen_idx >= 0 else 0.0
    return max(result, 0.0)

# Laplace
def laplace_sum(data, epsilon, U):
    true_sum = np.sum(data)
    noise = np.random.laplace(0, U / epsilon)
    return max(true_sum + noise, 0.0)

# cleaning
df_sp = pd.read_csv('Salaries.csv', low_memory=False)
df_sp['TotalPayBenefits'] = pd.to_numeric(df_sp['TotalPayBenefits'], errors='coerce')
df_sp = df_sp.dropna(subset=['TotalPayBenefits'])
data_sp = df_sp['TotalPayBenefits'].values
U_sp = 600000

df_ins = pd.read_csv('nyc-rolling-sales.csv')
df_ins['SALE PRICE'] = df_ins['SALE PRICE'].astype(str).str.replace(",", "").str.replace("$", "", regex=False)
df_ins['SALE PRICE'] = pd.to_numeric(df_ins['SALE PRICE'], errors='coerce')
df_ins = df_ins[df_ins['SALE PRICE'] > 0]
data_ins = df_ins['SALE PRICE'].values
U_ins = 2100000000

df_cc = pd.read_csv('creditcard.csv')
df_cc['Amount'] = pd.to_numeric(df_cc['Amount'], errors='coerce')
df_cc = df_cc.dropna(subset=['Amount'])
data_cc = df_cc['Amount'].values
U_cc = 25000

#
epsilons = [0.1, 0.5, 1, 2, 3, 4, 5]
runs = 50
results_frames = []

#
for dataset_name, data, U in [('Salaries', data_sp, U_sp),
                              ('nyc-rolling-sales', data_ins, U_ins),
                              ('creditcard', data_cc, U_cc)]:
    records = []
    true_sum = np.sum(data)
    for eps in epsilons:
        errors_sh, errors_lp = [], []
        for _ in range(runs):
            est_sh = sumdp_sum(data, eps, U)
            est_lp = laplace_sum(data, eps, U)
            errors_sh.append(est_sh - true_sum)
            errors_lp.append(est_lp - true_sum)
        errors_sh = np.array(errors_sh)
        errors_lp = np.array(errors_lp)
        records.append({
            'Dataset': dataset_name,
            'Epsilon': eps,
            'SumDP_MAE': np.mean(np.abs(errors_sh)),
            'SumDP_MSE': np.mean(errors_sh ** 2),
            'SumDP_STD': np.std(errors_sh),
            'Laplace_MAE': np.mean(np.abs(errors_lp)),
            'Laplace_MSE': np.mean(errors_lp ** 2),
            'Laplace_STD': np.std(errors_lp)
        })
    df_results = pd.DataFrame(records)
    results_frames.append(df_results)

    print(f"=== {dataset_name}  ===")
    for metric in ['MAE', 'MSE', 'STD']:
        pivot_table = df_results.pivot(index='Epsilon',
                                       columns='Dataset',
                                       values=[f'SumDP_{metric}', f'Laplace_{metric}'])
        pivot_table.columns = ['SumDP', 'Laplace']
        print(f"{metric}:")
        print(pivot_table)
    print()

#
for df_results in results_frames:
    dataset_name = df_results.loc[0, 'Dataset']
    eps_values = df_results['Epsilon'].values

    plt.figure(figsize=(12, 4))
    for i, metric in enumerate(['MAE', 'MSE', 'STD']):
        plt.subplot(1, 3, i + 1)
        plt.plot(eps_values, df_results[f'SumDP_{metric}'], marker='o', label='SumDP')
        plt.plot(eps_values, df_results[f'Laplace_{metric}'], marker='s', label='Laplace')
        plt.xlabel('ε')
        plt.ylabel(metric)
        plt.title(f'{dataset_name} {metric}')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()
