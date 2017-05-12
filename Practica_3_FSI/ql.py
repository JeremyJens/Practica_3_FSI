import random
import numpy as np
import matplotlib.pyplot as plt


# Environment size
width = 5
height = 16

# Actions
num_actions = 4

actions_list = {"UP": 0,
                "RIGHT": 1,
                "DOWN": 2,
                "LEFT": 3
                }

actions_list_inv = {0: "UP",
                1: "RIGHT",
                2: "DOWN",
                3: "LEFT"
                }

actions_vectors = {"UP": (-1, 0),
                   "RIGHT": (0, 1),
                   "DOWN": (1, 0),
                   "LEFT": (0, -1)
                   }

# Discount factor
discount = 0.8

Q = np.zeros((height * width, num_actions))  # Q matrix
Rewards = np.zeros(height * width)  # Reward matrix, it is stored in one dimension


def getState(y, x):
    return y * width + x


def getStateCoord(state):
    return int(state / width), int(state % width)


def getActions(state): #Devuelve una lista con las posibles acciones
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append("RIGHT")
    if x > 0:
        actions.append("LEFT")
    if y < height - 1:
        actions.append("DOWN")
    if y > 0:
        actions.append("UP")
    return actions


def getRndAction(state): #Nos devuelve un movimiento aleatorio
    return random.choice(getActions(state))


def getRndState(): #Nos devuelve un estado aleatorio
    return random.randint(0, height * width - 1)


Rewards[4 * width + 3] = -10000
Rewards[4 * width + 2] = -10000
Rewards[4 * width + 1] = -10000
Rewards[4 * width + 0] = -10000

Rewards[9 * width + 4] = -10000
Rewards[9 * width + 3] = -10000
Rewards[9 * width + 2] = -10000
Rewards[9 * width + 1] = -10000

Rewards[3 * width + 3] = 100
final_state = getState(3, 3)

print np.reshape(Rewards, (height, width))


def qlearning(s1, a, s2):
    Q[s1][a] = Rewards[s2] + discount * max(Q[s2])
    return

acciones = 0

def getGreedy(state):

    position = getStateCoord(state)

    position2 = getState(position[0],position[1])

    if (max(Q[position2]) == 0):
    #if (max(Q[state]) == 0):

        actions = getRndAction(state)

    else:

        best = np.argmax(Q[position2])
	#best = np.argmax(Q[state])

        actions = actions_list_inv[best]

    return actions

def get_E_Greedy(state, e):

    position = getStateCoord(state)

    position2 = getState(position[0],position[1])

    if (max(Q[position2]) == 0):
    #if (max(Q[state]) == 0):

        actions = getRndAction(state)

    elif(random.random() < e):

        best = np.argmax(Q[position2])
	#best = np.argmax(Q[state])

        actions = actions_list_inv[best]

    else:

        actions = getRndAction(state)

    return actions


# Episodes
elementos_random = []
elementos_greedy = []
elementos_e_greedy_1 = []
elementos_e_greedy_2 = []
episodios = 200

for i in xrange(episodios):
    accionesGr = 0
    state = getRndState()
    while state != final_state:
        acciones = acciones + 1
        accionesGr = accionesGr + 1
        action = getRndAction(state)
        y = getStateCoord(state)[0] + actions_vectors[action][0]
        x = getStateCoord(state)[1] + actions_vectors[action][1]
        new_state = getState(y, x)
        qlearning(state, actions_list[action], new_state)
        state = new_state

    if((i+1)%10 == 0):
        elementos_random.append(acciones/(i+1))

print "Random ", acciones/100

# Episodes
Q = np.zeros((height * width, num_actions))  # Q matrix
acciones = 0
for i in xrange(episodios):
    accionesGr = 0
    state = getRndState()
    while state != final_state:
        acciones = acciones + 1
        accionesGr = accionesGr + 1
        action = getGreedy(state)
        y = getStateCoord(state)[0] + actions_vectors[action][0]
        x = getStateCoord(state)[1] + actions_vectors[action][1]
        new_state = getState(y, x)
        qlearning(state, actions_list[action], new_state)
        state = new_state

    if ((i + 1) % 10 == 0):
        elementos_greedy.append(acciones/(i+1))

print "Greedy ", acciones/100

# Episodes
Q = np.zeros((height * width, num_actions))  # Q matrix
acciones = 0
for i in xrange(episodios):
    accionesGr = 0
    state = getRndState()
    while state != final_state:
        acciones = acciones + 1
        accionesGr = accionesGr + 1
        action = get_E_Greedy(state,0.9)
        y = getStateCoord(state)[0] + actions_vectors[action][0]
        x = getStateCoord(state)[1] + actions_vectors[action][1]
        new_state = getState(y, x)
        qlearning(state, actions_list[action], new_state)
        state = new_state

    if ((i + 1) % 10 == 0):
        elementos_e_greedy_1.append(acciones/(i+1))

print "E-Greedy-90 ", acciones/100

# Episodes
Q = np.zeros((height * width, num_actions))  # Q matrix
acciones = 0
for i in xrange(episodios):
    accionesGr = 0
    state = getRndState()
    while state != final_state:
        acciones = acciones + 1
        accionesGr = accionesGr + 1
        action = get_E_Greedy(state,0.95)
        y = getStateCoord(state)[0] + actions_vectors[action][0]
        x = getStateCoord(state)[1] + actions_vectors[action][1]
        new_state = getState(y, x)
        qlearning(state, actions_list[action], new_state)
        state = new_state

    if ((i + 1) % 10 == 0):
        elementos_e_greedy_2.append(acciones/(i+1))

print "E-greedy-95 ", acciones/100



# print Q


# Q matrix plot - Impresion de la tabla Q

s = 0
ax = plt.axes()
ax.axis([-1, width + 1, -1, height + 1])

for j in xrange(height):

    plt.plot([0, width], [j, j], 'b')
    for i in xrange(width):
        plt.plot([i, i], [0, height], 'b')

        direction = np.argmax(Q[s])
        if s != final_state:
            if direction == 0:
                ax.arrow(i + 0.5, 0.75 + j, 0, -0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 1:
                ax.arrow(0.25 + i, j + 0.5, 0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 2:
                ax.arrow(i + 0.5, 0.25 + j, 0, 0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
            if direction == 3:
                ax.arrow(0.75 + i, j + 0.5, -0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
        s += 1

    plt.plot([i+1, i+1], [0, height], 'b')
    plt.plot([0, width], [j+1, j+1], 'b')

plt.show()



def showActionsNumber(elementos_random,elementos_greedy,elementos_e_greedy_1,elementos_e_greedy_2):

    plt.plot(range(1,(episodios/10)+1), elementos_random, label='Explore')
    plt.plot(range(1,(episodios/10)+1), elementos_greedy, label='Greedy')
    plt.plot(range(1,(episodios/10)+1), elementos_e_greedy_1, label='E-Greedy 90')
    plt.plot(range(1,(episodios/10)+1), elementos_e_greedy_2, label='E-Greedy 95')
    plt.legend(loc='upper center', shadow=True)

    plt.axhline(0, color="black")
    plt.axhline(0, color="black")

    plt.show()

showActionsNumber(elementos_random,elementos_greedy,elementos_e_greedy_1,elementos_e_greedy_2)
