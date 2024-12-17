###############################################################################################
'''
The commander of your specialized team of agents. This agent is the most capable and can
utilize your team's strengths to complete the mission in the most effective and efficient
way possible.

Provide the commander with the roster of agents available, and watch as complex tasks
are broken down into sub-tasks that each agent is most proficient as.
'''
###############################################################################################
from MMO import Roster

class Commander():
    def __init__(self, roster: Roster, commander: str) -> None:
        self.roster = roster
        return None
    
    def getRoster(self) -> Roster:
        return self.roster

