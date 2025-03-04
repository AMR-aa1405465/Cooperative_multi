# POSSIBLE_BIT_RATES = [20,25,30,35,40]
# POSSIBLE_FRAME_RATES = [10,15,20,25,30]
# POSSIBLE_BEHAVIOURAL_ACCURACYS = [0.5,0.6,0.7,0.8,0.9]

# initial values. later I will change this to be more values. 
# POSSIBLE_BIT_RATES = [20,25,30,35,40]
# POSSIBLE_FRAME_RATES = [10,15,25,30]
# POSSIBLE_BEHAVIOURAL_ACCURACYS = [0.33,0.66,0.99]
UNIVERSAL_POSSIBLE_PERCENTAGES = [0.25,0.5,0.75,1]

class HeadType:
    LOCAL = 1    # a local head (within the msp domain)
    EXTERNAL = 2 # an external head (outside the msp domain)

VISUAL_QUALITY_WEIGHT = 0.4
RESPONSIVE_WEIGHT = 0.3
BEHAVIORAL_ACCURACY_WEIGHT = 0.3
NO_MONEY_SPENT_FLAG = -20

ACTION_DOABLE_AND_APPLIED = 1
ACTION_NOT_DOABLE = 0

PENALITY_WEIGHT = 10
MOVING_AVERAGE_WINDOW = 30
IMMERSIVNESS_FREEDOM = 0.98
PERC_TO_KEEP = 0.25 # the percentage of the budget that the msp should keep after helping.
LOWER_BUDGET_THRESHOLD = 0.25 # the threshold for the budget to be considered as low.


MIN_IMMERSIVENESS_TO_PASS = 0.6 # the minimum immersiveness to pass. 