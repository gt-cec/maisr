# TODO
## Priority
    * Modify logging to store final score and time remaining when game ends
    * Add word "HP" to health panels

## For later studies:
    * More intelligent rule based agent (e.g. maybe it will prioritize a single target far away if the human has the closer targets handled)

##  Lower priority
    * Implement code to draw next N target waypoints (started in autonomous policy class) (Code partially written in render but not updating properly
    * BUG: Game time not resetting when time done condition hit, so only the first game runs.
    * Intermittent location reporting
    * Implement holding patterns for hold policy and human when no waypoint set (currently just freezes in place)

##  Code optimization/cleanup
    * Move button handling code out of main.py and into isr_gui.py

#  Possible optimizations
  * Don't re-render every GUI element every tick. Just the updates
