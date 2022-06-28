import numpy as np
import numpy.random as rd
import random
from numba import vectorize, jit, cuda, float64
# project Durak
@jit(nopython=True)
def reset():
    deck = np.arange(52)
    rd.shuffle(deck)
    handp1 = np.zeros(52)
    for card in deck[:8]:
        handp1[card] =1
    handp2 = np.zeros(52)
    for card in deck[8:16]:
        handp2[card] =1 
    handp3 = np.zeros(52)
    for card in deck[16:24]:
        handp3[card] =1 
    handp4 = np.zeros(52)
    for card in deck[24:32]:
        handp4[card] =1 
    remain = deck[32:]
    last = np.array([deck[-1]])
    to_defense = np.array([-1])
    to_om = np.zeros(52) -1
    p_turn = np.zeros(1)
    handp1 = np.append(handp1,handp2)
    handp1 = np.append(handp1,handp3)
    handp1 = np.append(handp1,handp4)
    handp1 = np.append(handp1,p_turn)
    handp1 = np.append(handp1,last)
    handp1 = np.append(handp1,to_defense)
    handp1 = np.append(handp1,to_om)
    handp1 = np.append(handp1,remain)
    # env_state = handp1 + handp2 + handp3 + handp4 + p_turn + last + to_defense + to_om
    return handp1
env_state = reset()
env_state[263:]

def env_to_player(env_state):
    current_player = int(env_state[208]%4)
    hand = env_state[current_player*52:(current_player+1)*52]
    hand = np.append(hand,env_state[209:263])
    return hand
state = env_to_player(env_state)
# so la la bai danh, 53 la skip k danh nua
@jit(nopython = True)
def get_list_action(state):
    # check xem dang thu hay dang cong
    if state[53] < 0:
        # dang cong
        # check xem da danh la nao chua
        if np.sum(state[54:]) == 0:
            list_action = np.array([52])
            for act in range(52):
                if state[act] == 1:
                    list_action = np.append(list_action,act)
            return list_action
        # neu da danh
        else:
            # check xem cac la da danh ra la so nao
            da_danh = []
            for card in range(52):
                if state[54 + card] == 1:
                    da_danh.append(card//4)
            list_action = np.array([52])
            for card in range(52):
                if card//4 in da_danh and state[card] == 1:
                    list_action = np.append(list_action,card)
            return list_action
    else:
        # dang thu
        truong = state[52]%4
        # tinh diem la dang phai chan 
        number = state[53]//4
        suite = state[53]%4
        list_action = np.array([52])
        # neu doi thu danh truong
        if suite == truong:
            for card in range(52):
                if card%4 == truong and state[card] == 1 and card//4 > number:
                    list_action = np.append(list_action,card)
        else:
            for card in range(52):
                if state[card] == 1:
                    if card%4 == suite and card//4 > number:
                        list_action = np.append(list_action,card)
                    if card%4 == truong:
                        list_action = np.append(list_action,card)
        return list_action


state[53] = 1
print(state)
get_list_action(state)
# @jit(nopython=True)
def environment(env_state,choice):
    current_player = int(env_state[208]%4)
    to_def = int(env_state[208+1]%4)
    # khi la nguoi tan cong
    if env_state[210] < 0:
        # neu skip
        if choice == 52:
            # neu la nguoi cuoi cung duoc tan
            if env_state[210] == -3:
                # reset the phai om
                env_state[211:263] = 0
                # reset vi tri to_defense
                env_state[210] = -1
                # boc bai
                for player in range(4):
                    play = int((env_state[208] + player)%4)
                    if np.sum(env_state[play*52:(play+1)*52]) < 8: 
                        can_boc = 8 - np.sum(env_state[play*52:(play+1)*52])
                        for remainc in range(20):
                            if env_state[263 + remainc] > -1:
                                # thêm thẻ lên tay
                                env_state[int(env_state[263 + remainc] + play*52)] = 1
                                env_state[263 + remainc] = -1
                                can_boc -= 1
                                if can_boc == 0:
                                    break
                # them luot
                env_state[208] += 1
                return env_state
            else:
                # neu chua phai nguoi cuoi, nhuong luot cho nguoi khac
                env_state[210] -= 1
                return env_state
        # neu danh bai
        else:
            env_state[210] = choice
            env_state[choice + current_player*52] = 0
            env_state[choice + 211] = 1
            return env_state
    # khi la nguoi phong thu
    else:
        # neu skip
        if choice == 52:
            # om bai
            env_state[(1+current_player)*52:(2+current_player)*52] += env_state[211:263]
            # reset bai om
            env_state[211:263] = 0
            # reset bai chặn
            env_state[210] = -1
            # doi sang turn nguoi khac
            for player in range(4):
                play = int((env_state[208] + player)%4)
                if np.sum(env_state[play*52:(play+1)*52]) < 8:
                    can_boc = 8 - np.sum(env_state[play*52:(play+1)*52])
                    for remainc in range(20):
                        if env_state[263 + remainc] > -1:
                            # thêm thẻ lên tay
                            env_state[int(env_state[263 + remainc] + play*52)] = 1
                            env_state[263 + remainc] = -1
                            can_boc -= 1
                            if can_boc == 0:
                                break
            env_state[208] += 2
            return env_state
        # neu danh bai
        else:
            env_state[choice + (1+current_player)*52] = 0
            env_state[choice + 211] = 1
            env_state[210] = -1
            return env_state
environment(env_state,52)

@jit(nopython=True)
def check_win(env_state):
    for player in range(4):
        if np.sum(env_state[52*player:52*(player+1)]) == 0:
            return player
    return - 1


@jit(nopython=True)
def check_victory(state):
    if np.sum(state[:52]) == 0:
        return 1
    for other in range(1,4):
        if np.sum(state[52*other:52*(other+1)]) == 0:
            return 0
    return -1

@jit(nopython=True)
def amount_action_space():
    return 53

# @jit(nopython=True)
def player_random0(state,file_temp,file_per):
    a = get_list_action(state)
    b = random.randrange(len(a))
    return a[b],file_temp,file_per

def action_player(list_player,env_state,file_temp,file_per):
    state = env_to_player(env_state)
    play = int((env_state[208]+1)%4)
    if env_state[210] == -1:
        play = int(env_state[208]%4)
    if env_state[210] == -2:
        play = int((env_state[208]+2)%4)
    if env_state[210] == -3:
        play = int((env_state[208]+3)%4)
    choice,file_temp[play],file_per = list_player[play](state,file_temp[play],file_per)
    return choice,file_temp,file_per

def one_game(list_player,file_per):
    env_state = reset()
    file_temp = [[0],[0],[0],[0],[0]]
    # for turn in range(100):
    while check_win(env_state) == -1:
        # print(list_player,env_state,file_temp,file_per)
        choice,file_temp,file_per = action_player(list_player,env_state,file_temp,file_per)
        env_state = environment(env_state,choice)
        # print(env_state)
    state = env_to_player(env_state)
    for play in range(4):
        choice,file_temp[play],file_per = list_player[play](state,file_temp[play],file_per)
    return check_win(env_state),file_per

def normal_main(list_player,times,print_mode):
    count = [0,0,0,0]
    file_per = [0]
    list_randomed = [0,1,2,3]
    for van in range(times):
        rd.shuffle(list_randomed)
        shuffled_players = [list_player[list_randomed[0]],list_player[list_randomed[1]],list_player[list_randomed[2]],list_player[list_randomed[3]]]
        state = reset()
        win,file_per = one_game(shuffled_players,file_per)
        # print(turn)
        real_winner = list_randomed[win]
        count[real_winner] += 1
    return count,file_per
