# TODO
##  Priority 1:
    * Intermittent location reporting
    * Implement agent status window info in autonomous policy too
    * Show agent waypoint: 0 shows none, 1 shows next one, 2 shows next two, 3 shows next 3 (to be implemented inside agents.py
    * Populate agent_priorities (pull from autonomous policy)
    * Massively clean up agent policies. Make one default policy that avoids WEZs well but prioritizes badly.
    * Append subject ID (configurable here) to the log filename

##  Agent policies
    * (Priority) Target_id_policy: Currently a working but slow and flawed A* search policy is implemented
       (safe_target_id_policy). Have partially updated code in this script to replace target_id_policy but need to clean up.
    * Autonomous policy code shouldn't change quadrants until that quadrant is empty.

##  Lower priority
    * BUG: Game time not resetting when time done condition hit, so only the first game runs.
    * Implement holding patterns for hold policy and human when no waypoint set (currently just freezes in place)
    * Add waypoint command

##  Code optimization/cleanup
    * Move a lot of the button handling code out of main.py and into isr_gui.py

#  Possible optimizations
  * Don't re-render every GUI element every tick. Just the updates