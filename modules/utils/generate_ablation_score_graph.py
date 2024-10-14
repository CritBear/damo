
import numpy as np
import matplotlib.pyplot as plt

# x축 데이터
category = ['Synthetic\n( J )', 'Synthetic\n( G, J )', 'Synthetic\n( O, G, J )', 'Synthetic\n( O, G, J, S )', 'Real']

subjects = ['Baseline', 'Cross to Self', 'RJ to AJ']

eval = 'JPE'

JPE_data = [
    [24.24, 24.08, 24.15, 24.10, 42.56],
    [24.24, 24.08, 24.15, 24.10, 42.56],
    [24.24, 24.08, 24.15, 24.10, 42.56]
]

JOE_data = [
    [19.30, 18.76, 18.40, 18.46, 24.37],
    [],
    []
]

# y축 데이터 (오류의 정도, deg 단위)

colors = ['#7ed956', '#888888', '#333333']

if eval == 'JPE':
    data = JPE_data
else:
    data = JOE_data

# x축 좌표
x = np.arange(len(category))

# 막대 그래프 폭
bar_width = 0.15

# 색깔
fig, ax = plt.subplots() # figsize=(7, 12)
plt.rcParams['font.family'] = 'serif'

x_offset = [-bar_width, 0, bar_width]

for i in range(3):
    plt.bar(x + x_offset[i], data[i], width=bar_width, label=subjects[i], color=colors[i])

# x축 눈금 및 레이블 설정
plt.xticks(x, category)
plt.yticks()
# ax.tick_params(axis='x', pad=60)
# ax.tick_params(axis='y', labelsize=22)

# plt.rcParams.update({'font.size': 22})

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