// calculate the current reward of the state 
function calculateReward() {
    // reward for being close to ships
    distanceReward = 0
    targetClassificationReward = 0
    threatClassificationReward = 0
    // timeReward = -1 * timeElapsed * 0.1  // -0.1 per second

    for (let i = 0; i < minDistancesToShips.length; i++) {
        distanceReward += (initialDistancesToShips[i] - minDistancesToShips[i]) / initialDistancesToShips[i]  // fractional progress for each ship, each ship is 1 reward
        targetClassificationReward += seenTargetClasses[i] ? 1 : 0
        threatClassificationReward += seenThreatClasses[i] ? 1 : 0
    }

    // get dist to closest non-classified ship
    let minDist = 99999
    let closestShip = null
    for (let i = 0; i < targetShips.length; i++) {
        let dist = distance(targetShips[i].x, aircraft1.x, targetShips[i].y, aircraft1.y)
        if (dist < minDist) {
            minDist = dist
            closestShip = i
        }
    }

    if (closestShip == null) {
        minDist = 0
    }

    minDist = -1 * pxToUnaryW(minDist)  // negative so dist subtracts from the reward

    // console.log("Reward:", distanceReward, targetClassificationReward, threatClassificationReward, timeReward, "=", distanceReward + targetClassificationReward + threatClassificationReward + timeReward)
    
    return distanceReward + targetClassificationReward + threatClassificationReward + minDist  //+ timeReward
}