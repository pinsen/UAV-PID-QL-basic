import numpy as np
import matplotlib.pyplot as plt
import random

dy = [1,0,-1,0]
dx = [0,1,0,-1]

grid_size = 5
num_states = grid_size*grid_size
num_actions = 4

def cord_to_state(x,y):
    return x*grid_size + y
def state_to_cord(s):
    return s//grid_size, s%grid_size

# START / GOAL STATES
start = [0,0]
goal = [4,4]

# konstan / number episode bisa dioatak atik
alpha = 0.5
gamma = 0.7
epsilon = 0.1
n = 500 # <-- yg ini number episode
# konstannya pid
Kp = 0.8
Kd = 0.9
Ki = 0.0    
d = 0.3 # <-- error radius
dt = 0.01 # <-- interval

Q = np.zeros((num_states, num_actions))
steps_per_episode = np.zeros(n)

for ep in range(n):
    cur_s = cord_to_state(start[0],start[1])
    counter_step = 0
    while True:
        y,x = state_to_cord(cur_s)
        action = 0
        if(random.random()<epsilon):
            action = random.randint(0,num_actions-1)
        else:
            maks = np.max(Q[cur_s,:])
            actions = np.where(Q[cur_s,:]==maks)[0]
            action = random.choice(actions)

        next_y = y + dy[action]
        next_x = x + dx[action]

        if(next_x < 0 or next_x >= grid_size or next_y < 0 or next_y >= grid_size):
            next_y, next_x = y,x

        next_s = cord_to_state(next_y,next_x)

        target = np.array([next_y,next_x],dtype=float)
        cur_pos = np.array([y,x],dtype=float)
        prev_eror = target - cur_pos

        integral = 0
        while True:
            error = target - cur_pos

            if np.linalg.norm(error) <= d:
                break

            derivative = (error - prev_eror) / dt
            integral += error*dt
            prev_eror = error

            ut = Kp*error + Kd*derivative + Ki*integral

            cur_pos += ut*dt

        if next_s == cord_to_state(goal[0],goal[1]):
            reward = 100
        else:
            reward = -1

        Q[cur_s, action] = (1-alpha)*Q[cur_s,action] + alpha*(reward + gamma*np.max(Q[next_s,:]))

        cur_s = next_s
        counter_step+=1
        steps_per_episode[ep] = counter_step
        if counter_step>500 or cur_s == cord_to_state(goal[0],goal[1]):
            break

plt.figure(facecolor='white')
plt.plot(range(1, n + 1), steps_per_episode, 'b-', linewidth=1.5)
plt.grid(True)
plt.xlabel('Episode')
plt.ylabel('Number of Steps')
plt.title('Steps Taken Per Learning Episode')
plt.xlim([0, n])
plt.ylim([0, np.max(steps_per_episode) + 5])

print("\n--- Final Operational Metrics ---")
print(f"Langkah awal episode 1: {int(steps_per_episode[0])} steps")
print(f"Langkah akhir saat konvergen (Jalur Terpendek): {int(steps_per_episode[-1])} steps")

plt.show()