import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.linalg import solve
from scipy.stats import truncnorm

plt.rcParams['font.family'] = 'sans-serif'

# 固定パラメータ
dx = dy = 1 
dt = 0.0005 
Nt = 1000  
border = 1
VacantSpace = 1
N = 25
Organism = 1
mean = 2.21
std = 0.96
frame_interval = 10

def build_periodic_tridiagonal(N, r):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, i] = 1 + 2 * r
        A[i, (i - 1) % N] = -r
        A[i, (i + 1) % N] = -r
    return A

def adi_step_periodic(u, D, dt, dx, dy):
    rx = D * dt / (2 * dx**2)
    ry = D * dt / (2 * dy**2)

    A_x = build_periodic_tridiagonal(N, rx)
    u_half = np.zeros_like(u)

    for j in range(N):
        rhs = u[j, :] + ry * (np.roll(u[(j+1)%N, :], 0) - 2*u[j, :] + np.roll(u[(j-1)%N, :], 0))
        u_half[j, :] = solve(A_x, rhs)

    A_y = build_periodic_tridiagonal(N, ry)
    u_next = np.zeros_like(u)

    for i in range(N):
        rhs = u_half[:, i] + rx * (np.roll(u_half[:, (i+1)%N], 0) - 2*u_half[:, i] + np.roll(u_half[:, (i-1)%N], 0))
        u_next[:, i] = solve(A_y, rhs)

    return u_next

def ingredient(data, lag_timer):
    data_new = np.copy(data)
    lag_timer_new = np.copy(lag_timer)
    lag_timer_new[data[1] >= 1] += 1
    
    for i in range(N):
        for j in range(N):
            if data[1, i, j] >= 1:
                neighbors = [
                    ((i-1)%N, j),
                    ((i+1)%N, j),
                    (i, (j-1)%N),
                    (i, (j+1)%N)
                ]

                around_nutrient = [(data[0, ni, nj], (ni, nj)) for (ni, nj) in neighbors]
                total_nutrient = sum(val for val, _ in around_nutrient)
                empty_spaces = [(ni, nj) for (ni, nj) in neighbors if data_new[1, ni, nj] == 0]

                if total_nutrient >= border and len(empty_spaces) >= VacantSpace:
                    a, b = (0 - mean) / std, (np.inf - mean) / std
                    sampled_lag = truncnorm.rvs(a, b, loc=mean, scale=std) * 60
                    if sampled_lag < lag_timer[i, j]:
                        around_nutrient.sort(reverse=True, key=lambda x: x[0])
                        p = np.random.rand() * total_nutrient
                        cumulative = 0
                        for nutrient, (ni, nj) in around_nutrient:
                            cumulative += nutrient
                            if p < cumulative and (ni, nj) in empty_spaces:
                                need = border
                                for val, (mi, mj) in around_nutrient:
                                    consume = min(val, need)
                                    data[0, mi, mj] -= consume
                                    data_new[0, mi, mj] -= consume
                                    need -= consume
                                    if need <= 0:
                                        break
                                data_new[1, ni, nj] += 1
                                lag_timer_new[i, j] = 0
                                lag_timer_new[ni, nj] = 0
                                break
    return data_new, lag_timer_new

def run_simulation(D, Nutrition):
    ing = np.zeros((N, N), dtype=np.int64)
    nums = np.random.choice(N*N, size=int(Nutrition))
    y = nums // N
    x = nums % N
    np.add.at(ing, (y, x), 1)
    ing = ing.astype(np.float64)

    org = np.zeros((N, N), dtype=np.int64)
    org[N//2, N//2] = Organism

    data = np.array([ing, org])
    lag_timer = np.zeros_like(data[1])
    
    frames = []
    norm1 = Normalize(vmin=0, vmax=1)
    norm2 = Normalize(vmin=0, vmax=1)

    for t in range(Nt):
        data[0] = adi_step_periodic(data[0], D, dt, dx, dy)
        if t % frame_interval == 0:
           if t % (frame_interval*50) == 0:
              fig, ax = plt.subplots(figsize=(3, 3))
              ax.imshow(data[0], cmap='gray_r', norm=norm1)
              ax.imshow(data[1], cmap='Reds', alpha=0.5, norm=norm2)
              ax.text(5, -1, '赤色：細菌（さいきん）', fontsize=8)
              ax.text(15, -1, 'グレー：栄養（えいよう）', fontsize=8)
              ax.axis("off")
              frames.append(fig)
           data, lag_timer = ingredient(data, lag_timer)
    return frames

def main():
    st.title("細菌（さいきん）がふえる様子を見てみよう！")
    # 対数スケールの値をリストで作成
    D_values = [0, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    D = st.select_slider('栄養（えいよう）の広がりやすさ', options=D_values, value=D_values[3])
    Nutrition_values = list(range(0, 1001, 50))
    Nutrition = st.select_slider("栄養（えいよう）の量（りょう）", options=Nutrition_values, value=Nutrition_values[10])

    if "frames" not in st.session_state:
        st.session_state.frames = None

    if st.button("シミュレーション スタート!"):
        st.info("シミュレーション中...ちょっと待ってね")
        frames = run_simulation(D, Nutrition)
        st.session_state.frames = frames
        st.success("細菌（さいきん）はこんなふうにふえるよ↓")

    if st.session_state.frames:
        step = st.slider("フレーム", 0, len(st.session_state.frames) - 1, 0)
        st.pyplot(st.session_state.frames[step], use_container_width=False)

if __name__ == "__main__":
    main()
