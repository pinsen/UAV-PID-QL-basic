import numpy as np
# import matplotlib.pyplot as plt
import random
import socket  # <-- PUSTAKA BARU UNTUK KIRIM DATA
import time    # <-- PUSTAKA BARU UNTUK JEDA VISUAL

# Konfigurasi Pemancar Sinyal ke Unity
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

dy = [1, 0, -1, 0]
dx = [0, 1, 0, -1]

grid_size = 10
num_states = grid_size * grid_size
num_actions = 4

def cord_to_state(x, y):
    return x * grid_size + y

def state_to_cord(s):
    return s // grid_size, s % grid_size

start = [0, 0]
goal = [9, 9]

alpha = 0.5
gamma = 0.7
epsilon = 0.1
n = 1000 

Q = np.zeros((num_states, num_actions))
steps_per_episode = np.zeros(n)

print("Memulai Latihan Q-Learning + Pemancaran Visualisasi ke Unity...")

for ep in range(n):
    cur_s = cord_to_state(start[0], start[1])
    counter_step = 0
    
    while True:
        y, x = state_to_cord(cur_s)

        message = f"{y},{x}"
        sock.sendto(bytes(message, "utf-8"), (UDP_IP, UDP_PORT))
        time.sleep(0.05)
        
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            maks = np.max(Q[cur_s, :])
            actions = np.where(Q[cur_s, :] == maks)[0]
            action = random.choice(actions)

        next_y = y + dy[action]
        next_x = x + dx[action]

        if next_x < 0 or next_x >= grid_size or next_y < 0 or next_y >= grid_size:
            next_y, next_x = y, x

        next_s = cord_to_state(next_y, next_x)

        if next_s == cord_to_state(goal[0], goal[1]):
            reward = 100
        else:
            reward = -1

        Q[cur_s, action] = (1 - alpha) * Q[cur_s, action] + alpha * (reward + gamma * np.max(Q[next_s, :]))

        cur_s = next_s
        counter_step += 1
        
        if counter_step > 500 or cur_s == cord_to_state(goal[0], goal[1]):

            sock.sendto(bytes(f"{next_y},{next_x}", "utf-8"), (UDP_IP, UDP_PORT))
            break

    steps_per_episode[ep] = counter_step