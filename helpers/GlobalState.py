class GlobalState:
    """
    Utility class to manage shared state across MSPs
    """
    clock = 0
    
    @classmethod
    def increment_clock(cls):
        cls.clock += 1
        
    @classmethod
    def reset_clock(cls):
        cls.clock = 0 

    @classmethod
    def get_clock(cls):
        return cls.clock

