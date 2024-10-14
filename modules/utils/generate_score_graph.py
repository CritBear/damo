
import numpy as np
import matplotlib.pyplot as plt

# x축 데이터
category_SOMA = ['Synthetic\n( J )', 'Synthetic\n( G, J )', 'Synthetic\n( O, G, J )', 'Synthetic\n( O, G, J, S )', 'Real']
category_MS = ['Labeled\nSynthetic\n( J )', 'Labeled\nSynthetic\n( Sh, J )', 'Labeled\nSynthetic\n( O, Sh, J )', 'Labeled\nSynthetic\n( O, Sh, J, Sw )']

dataset = 'SOMA'
eval = 'JPE'

DAMO_JPE = [38.31, 39.02, 41.15, 43.31, 35.54]
DAMO_JOE = [18.30, 18.76, 20.40, 22.46, 15.8]
SOMA_JPE = [71.71, 67.0, 299.47, 566.67, 44.0]
SOMA_JOE = [23.99, 22.52, 64.82, 80.66, 12.96]

DAMO_MS_JPE = [41.19, 42.43, 43.25, 43.75]
DAMO_MS_JOE = [20.92, 20.34, 20.97, 21.25]
MS_JPE = [12.89, 22.62, 122.59, 123.65]
MS_JOE = [5.55, 8.40, 29.92, 30.66]

# y축 데이터 (오류의 정도, deg 단위)

color_damo = '#7ed956' #'#0072bd'
color_soma = '#003f72' #'#edb120'
color_mocap_solver = '#009299' #'#d64d4d'

if dataset == 'SOMA':
    category = category_SOMA
    label = 'SOMA'
    color = color_soma
    if eval == 'JPE':
        error_deg_1 = DAMO_JPE
        error_deg_2 = SOMA_JPE
    else:
        error_deg_1 = DAMO_JOE
        error_deg_2 = SOMA_JOE
else:
    category = category_MS
    color = color_mocap_solver
    label = 'MoCap-Solver'
    if eval == 'JPE':
        error_deg_1 = DAMO_MS_JPE
        error_deg_2 = MS_JPE
    else:
        error_deg_1 = DAMO_MS_JOE
        error_deg_2 = MS_JOE

category = category[:len(error_deg_1)]

# x축 좌표
x = np.arange(len(category))

# 막대 그래프 폭
bar_width = 0.35

# 색깔
fig, ax = plt.subplots(figsize=(7, 12))
plt.rcParams['font.family'] = 'serif'

# 첫 번째 막대 그래프 그리기 (파란색)
plt.bar(x - bar_width/2, error_deg_1, width=bar_width, label='DAMO', color=color_damo)  # , edgecolor='black'

# 두 번째 막대 그래프 그리기 (주황색)
plt.bar(x + bar_width/2, error_deg_2, width=bar_width, label=label, color=color)

# x축 눈금 및 레이블 설정
plt.xticks(x, category, fontsize=22, rotation=45, ha='center', va='center')
plt.yticks()
ax.tick_params(axis='x', pad=60)
ax.tick_params(axis='y', labelsize=22)

plt.rcParams.update({'font.size': 22})

if eval == 'JPE':
    plt.ylim(0, 150)
else:
    plt.ylim(0, 50)
# plt.rcParams.update({'font.size': 12})
# 그래프에 제목 추가
# plt.title(f"DAMO vs SOMA ({eval})")

# x축 레이블 추가


# plt.xlabel("Noise")

# y축 레이블 추가
if eval == "JPE":
    error = "Joint Position Error"
    unit = 'mm'
else:
    error = "Joint Orientation Error"
    unit = 'deg'

# plt.ylabel(f"{eval} ({unit})")
plt.title(f"{error} ({unit})")

# 범례 추가
plt.legend()

# 그래프 표시
plt.tight_layout()
plt.show()