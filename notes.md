# Notes

1. The MSP is the one gets helped by other MSPs. 
2. The MSP will get a help percentage from the other MSPs.
3. each msp will have mutliple heads


* An Action is given to each MSP, 
* Each MSP would have the same number of heads inside of it. 
* An action is given to each MSP like 

DRL action to MSP 0 -> 3 , 3 will be decoded as [[bitrate,framerate,behavoural-accuracy],[bitrate,framerate,behavoural-accuracy]] for the 2 Heads the MSP has. 
if the head had 3 heads, then the action would be for all of them 

the number of configuration depends on the number of heads the msp has. 

# i will think about this one. 
def calculate_final_reward(self):
    # Combine multiple terminal reward components
    
    # 1. Budget efficiency
    remaining_ratio = sum(msp.get_budget() for msp in self.msp_list) / sum(msp.initial_budget for msp in self.msp_list)
    budget_reward = -abs(remaining_ratio - 0.1) * 10
    
    # 2. Average immersiveness achieved
    avg_immersiveness = sum(self.episode_overall_avg_imrvnss_lst) / len(self.episode_overall_avg_imrvnss_lst)
    immersiveness_reward = avg_immersiveness * 15
    
    # 3. Survival reward
    survival_ratio = GlobalState.get_clock() / self.max_clock
    survival_reward = survival_ratio * 20
    
    # 4. Consistency reward (low variance in immersiveness)
    immersiveness_variance = np.var(self.episode_overall_avg_imrvnss_lst)
    consistency_reward = -immersiveness_variance * 10
    
    final_reward = budget_reward + immersiveness_reward + survival_reward + consistency_reward
    return final_reward