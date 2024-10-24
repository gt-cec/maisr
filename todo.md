# TODO
##  Priority 1: Overhaul the agent info feature.
    * Append Type rationale and area rationale to the same line as type/area in parentheses
        * Search type: WEZ (Prioritize mission)
        * Search type: Target (Preserve health)
        * Search area: Full
        * Search area: NW (Grouping detected)

    * Brainstorm how to make agent status rationale useful (Maybe the above is good enough)
    * Make game window size configurable so it fits on other monitors
    * Fix time at game end (not resetting correctly) 
    * make it launch a qualtrics online survey

After priority 1: Type up description and send to lab

##  Lower priority
    * Implement code to draw next N target waypoints (started in autonomous policy class) (Code partially written in render but not updating properly
    * BUG: Game time not resetting when time done condition hit, so only the first game runs.
    * Intermittent location reporting
    * Implement holding patterns for hold policy and human when no waypoint set (currently just freezes in place)

##  Code optimization/cleanup
    * Move a lot of the button handling code out of main.py and into isr_gui.py

#  Possible optimizations
  * Don't re-render every GUI element every tick. Just the updates
