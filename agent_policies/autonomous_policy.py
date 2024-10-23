#Agent0 always runs autonomous policy.Buttons force specific constraints to behavior

class Policy:
    def __init__(self):


def aggressive_policy():
    ID
    target + WEZ


def cautious_policy():
    ID
    target
    only


def collision_avoidance_policy():
    If
    hostile is within
    X
    pixels
    straight
    ahead, command
    a
    deflected
    waypoint
    off
    to
    the
    side
    Find
    way
    to
    handle
    being
    surrounded


def upcoming_collision():
    pass

'''
In main.py, will need to hook up the buttons to set autonomous_policy.gameplan, .risk_tolerance, .search_quadrant, .target_priorities
'''

# TODO left off here. Should decide how the agent picks its quadrants, target priorities

class AutonomousPolicy:
    def __init__(self):
        self.low_level_rationale = '' # Identify unknown target, identify unknown WEZ, or evade threat
        self.high_level_rationale = ''
        self.search_quadrant = 'auto'  # 'auto' by default, NW/SW/NE/SE if human clicks a quadrant. Resets back to auto if autonomous button clicked
        self.target_priorities = 'auto'  # 'target' or 'wez' if human clicks buttons. Resets to auto if auto button clicked
        self.gameplan = 'auto'  # Overrides with 'cautious' or 'aggressive' if human clicks a gameplan button
        self.collision_ok = False  # If False, collision avoidance executes normally.
        self.risk_tolerance = 'medium'  # Override to low/high based on button clicks


    def calculate_risk_level(self):
        return self.risk_level

    def calculate_priorities(self):
        if self.risk_tolerance == 'low':
            self.gameplan = 'target'
            self.collision_okay = False
        if self.risk_tolerance == 'medium':
            self.gameplan = 'auto'
            self.collision_okay = False
        if self.risk_tolerance == 'high':
            self.gameplan = 'wez'
            self.collision_okay = False


    def choose(self):
        self.calculate_priorities()

        if upcoming_collision() and not self.collision_ok:
            collision_avoidance_policy()
            low_level_rationale = 'EVADE THREAT'

        # TODO think through this flow.
        #if self.risk_tolerance == 'medium':


        elif self.risk_level == low or self.risk_level == medium:
            aggressive_policy():
            high_level_rationale = 'Prioritize targets'
        elif self.risk_level == high or self.risk_level == extreme:
            cautious_policy():
            high_level_rationale = 'Preserve health'