

"""
The file contains utility functions for the project.
"""


from Constants import POSSIBLE_BIT_RATES,POSSIBLE_FRAME_RATES,POSSIBLE_BEHAVIOURAL_ACCURACYS

def msp_action_generator(bit_rates:list[float]=POSSIBLE_BIT_RATES,frame_rates:list[float]=POSSIBLE_FRAME_RATES,behavioural_accuracys:list[float]=POSSIBLE_BEHAVIOURAL_ACCURACYS):
    """
    Generates the list of possible actions for each MSP.
    """
    possible_actions = []
    for bit_rate in bit_rates:
        for frame_rate in frame_rates:
            for behavioural_accuracy in behavioural_accuracys:
                action = [bit_rate, frame_rate, behavioural_accuracy]
                possible_actions.append(action)
    return dict(enumerate(possible_actions))




if __name__ == "__main__":
    p = msp_action_generator(2)
    p_len = len(p)
    print(p[0])
    print(p[1])
    print(p[2])
    print(p[3])
    print(p[4])
    print(p[5])
    print(p[6])
    print(p[7])
    print(p[8])
    print(p[p_len-1])
