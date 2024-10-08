
// Game letiables
let targetShips;
let isAllCombatantsInvisibleSent;
let isAllTargetsInvisibleSent;
let phase1Complete;
let phase2Complete;
let phase3Complete;
let usprAlertsCounter;
var searchCompletions;
// counter for Phase 3 Tracking 
var minClassificationCounter;
// optimized search parameters
let shift_east;
let shift_south;
let numOptimized = 0;
let wez_alerts = 0;
let osi_alerts = 0;
// gain for MSFS speed
let speedGain

// Main game function
function init(numTargetsIteration, targetMotionIteration, searchPattern) {
    isAllCombatantsInvisibleSent = false;
    isAllTargetsInvisibleSent = false;
    phase1Complete = false;
    phase2Complete = false;
    phase3Complete = false;
    usprAlertsCounter = 0;
    searchCompletions = 0;
    minClassificationCounter = 0;
    // set ship speed gain for MSFS realistic scenario
    speedGain = 1;
    if (usingMSFS){
        speedGain = 0.1
    }

    /* Initialize User Aircraft */
    let aircraft1Path;
    let aircraft2Path;
    let nogoOffset = 0;
    if (gameplayColor === "yellow" || gameplayColor === "red") {
        nogoOffset = 0.2;
    }
    
    switch (searchPattern.toLowerCase()) {
        case "ladder":
            aircraft1Path = new WaypointPath([
                toSurv(1 - nogoOffset, 0/3),
                toSurv(1 - nogoOffset, 1/3),
                toSurv(0, 1/3),
                toSurv(0, 2/3),
                toSurv(1 - nogoOffset, 2/3),
                toSurv(1 - nogoOffset, 3/3),
                toSurv(0, 3/3),
            ]);
            aircraft2Path = new WaypointPath([
                toSurv(0, 3/3),
                toSurv(1 - nogoOffset, 3/3),
                toSurv(1 - nogoOffset, 2/3),
                toSurv(0, 2/3),
                toSurv(0, 1/3),
                toSurv(1 - nogoOffset, 1/3),
                toSurv(1 - nogoOffset, 0/3),
            ]);
        break;
        case "hold":
            aircraft1Path = new WaypointPath([
                toSurv(3/4, 0),
                toSurv(3/4, 1),
                toSurv(1/4, 1),
                toSurv(1/4, 0),
            ]);
            aircraft2Path = new WaypointPath([
                toSurv(1/4, 0),
                toSurv(1/4, 1),
                toSurv(3/4, 1),
                toSurv(3/4, 0),
            ]);
        break;
        case "square":
            aircraft1Path = new WaypointPath([
                toSurv(2/3, 2/4),
                toSurv(2/3, 3/4),
                toSurv(1/3, 3/4),
                toSurv(1/3, 1/4),
                toSurv(3/3 - nogoOffset, 1/4),
                toSurv(3/3 - nogoOffset, 4/4),
                toSurv(0/3, 4/4),
                toSurv(0/3, 0/4),
                toSurv(3/3 - nogoOffset, 0/4),
            ]);
            aircraft2Path = new WaypointPath([
                toSurv(3/3 - nogoOffset, 0/4),
                toSurv(0/3, 0/4),
                toSurv(0/3, 4/4),
                toSurv(3/3 - nogoOffset, 4/4),
                toSurv(3/3 - nogoOffset, 1/4),
                toSurv(1/3, 1/4),
                toSurv(1/3, 3/4),
                toSurv(2/3, 3/4),
                toSurv(2/3, 2/4),
            ])
        break
    }

    let aircraft1Start = toSurv(0, 1);
    let aircraft2Start = toSurv(1, 0);

    aircraft1 = new UserAircraft("aircraft 1", aircraft1Start.x, aircraft1Start.y, aircraft1Path);
    aircraft2 = new UserAircraft("aircraft 2", aircraft2Start.x, aircraft2Start.y, aircraft2Path);

    // RL: aircraft positions 
    currentObservation["observation"]["agent1 pos"] = [pxToUnaryW(aircraft1.x), pxToUnaryH(aircraft1.y)]
    currentObservation["observation"]["agent2 pos"] = [pxToUnaryW(aircraft2.x), pxToUnaryH(aircraft2.y)]

    // set ship speed gain for sim or MSFS realistic scenario
    // scale to sim aircraft relative speed userAircraft.speed / speedGain = AircraftSpeedKts Kts / 1kt
    let simmedAircraftSpeedKts = 100.0; 
    let MSFSAircraftSpeedKts = 270.0;
    speedGain = aircraft1.speed / simmedAircraftSpeedKts;
    if (usingRL) {
        speedGain = aircraft1.speed / 20.0;
    }
    if (usingMSFS){
        speedGain = aircraft1.speed / MSFSAircraftSpeedKts;
    }

    /* Initialize targets */
    targetShips = [];
    // Get the number of targets
    let numTargets;
    switch (numTargetsIteration) {
        case "A": numTargets = 10; break;
        case "B": numTargets = 20; break;
        case "C": numTargets = 30; break;
        case "D": numTargets = 50; break;
        case "E": numTargets = 100; break;
    }

    // RL: reset the threat observations
    currentObservation["observation"]["threat pos"] = []
    currentObservation["observation"]["threat class"] = []
    currentObservation["terminated"] = false
    seenThreatClasses = []
    seenTargetClasses = []
    
    // Create the targets
    let usedTargetIds = new Set();
    
    for (let i = 0; i < numTargets; i++) {
        // Get a random x,y
        let x = zeroX + rand()*gameWidth;
        let y = zeroY + rand()*gameHeight;

        // RL: save the threat position
        currentObservation["observation"]["threat pos"].push([pxToUnaryW(x), pxToUnaryH(y)])
        currentObservation["observation"]["threat class"].push([1, 0, 0, 0])  // unknown class
        initialDistanceAircraft1 = distance(x, aircraft1.x, y, aircraft1.y)
        initialDistanceAircraft2 = distance(x, aircraft2.x, y, aircraft2.y)
        minDist = initialDistanceAircraft1 < initialDistanceAircraft2 ? initialDistanceAircraft1 : initialDistanceAircraft2
        initialDistancesToShips.push(minDist)
        minDistancesToShips.push(minDist)
        seenTargetClasses.push(false)
        seenThreatClasses.push(false)

        // Get a random speed according to iteration
        let speed;
        let r;
        switch (targetMotionIteration) {
            case "F": speed = 0; break;
            case "G": speed = 5; break;
            case "H": speed = 10; break;
            case "I": speed = 15; break;
            case "J":
                r = rand();
                switch (true) {
                    case (r <= 0.5): speed = 0; break;
                    case (r <= 0.8): speed = 5; break;
                    case (r <= 0.95): speed = 10; break;
                    case (r <= 1): speed = 15; break;
                }
            break;
            case "K":
                r = rand();
                switch (true) {
                    case (r <= 0.5): speed = 15; break;
                    case (r <= 0.8): speed = 10; break;
                    case (r <= 0.95): speed = 5; break;
                    case (r <= 1): speed = 0; break;
                }
            break;
        }

        // Get random target id
        let targetId;
        while (1) {
            targetId = Math.floor(rand()*(10000-1000) + 1000);

            // Ensure that multiple targets do not have the same id
            if (!usedTargetIds.has(targetId)) {
                usedTargetIds.add(targetId);
                break;
            }
        }

        // Get the target class
        let targetClass = Math.floor(rand()*2);

        // Get the threat class
        let threatClass;
        if (targetClass === 0) {
            threatClass = 0;
        } else {
            threatClass = Math.floor(rand()*(4-1) + 1)
        }

        // Create new target
        newTarget = new TargetShip(x, y, speed*speedGain, targetId, targetClass, threatClass);
        newTarget.setRandomWaypoint();
        targetShips.push(newTarget)
    }

    // send ship info and waypoints to webserver route /ships
    // 'ships' param structured as:
    // 
    // [
    //   [  # ship
    //      [ID, targetClass, threatClass]
    //      [x, y, s],  # waypoint
    //      [x, y, s],
    //      [x, y, s],
    //      ...
    //   ],
    //   ...
    //  ]
    let data = []
    for (let i = 0; i < numTargets; i++)
    {
        data.push([])
        data[i].push([])
        data[i][0].push(targetShips[i].id)
        data[i][0].push(targetShips[i].targetClass)
        data[i][0].push(targetShips[i].threatClass)
        for(let j = 0; j < 30; j++)
        {
            data[i].push([])
            data[i][j+1].push((targetShips[i].waypointArrayX[j] - zeroX) / gameWidth)
            data[i][j+1].push((targetShips[i].waypointArrayY[j] - zeroY)/ gameHeight)
            data[i][j+1].push(targetShips[i].speed / speedGain)
        }
    }

    fetch('/ships', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({"ships": data})
    })

    // send initial aircraft destination
    fetch('/current-destination', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "aircraft 1 dest x": pxToUnaryW(aircraft1.path.currentWaypoint.x),
            "aircraft 1 dest y": pxToUnaryH(aircraft1.path.currentWaypoint.y),
            "aircraft 2 dest x": pxToUnaryW(aircraft2.path.currentWaypoint.x),
            "aircraft 2 dest y": pxToUnaryH(aircraft2.path.currentWaypoint.y)
        })
    })
}

// Game loop
function run(ctx) {
    if (!hasStarted) {
        return;
    }

    // clear the occupancy grid2d
    grid2d = new Array(gridHeight)
    for (let i = 0; i< gridHeight; i++){grid2d[i] = new Array(gridWidth).fill(1)}
   
    // populate occupancy grid at aircraft position on grid
    ac1Grid = pxToGrid(aircraft1.x, aircraft1.y)
    ac2Grid = pxToGrid(aircraft2.x, aircraft2.y)
    grid2d[ac1Grid.y][ac1Grid.x] = 0;
    grid2d[ac2Grid.y][ac2Grid.x] = 0;

    // find grid coordinates of next aircraft waypoint
    ac1NextWPGrid = pxToGrid(aircraft1.path.currentWaypoint.x, aircraft1.path.currentWaypoint.y)
    if (aircraft1Vector){
        ac1NextWPGrid = pxToGrid(aircraft1.nextWaypoint.x, aircraft1.nextWaypoint.y)
    }

    ac2NextWPGrid = pxToGrid(aircraft2.path.currentWaypoint.x, aircraft2.path.currentWaypoint.y)
    if (aircraft2Vector){
        ac2NextWPGrid = pxToGrid(aircraft2.nextWaypoint.x, aircraft2.nextWaypoint.y)
    }

    // Animate targets
    let isAllTargetsInvisible = true
    let isAllCombatantsInvisible = true

    let mouseOverTarget = null  // holding variable for a target that is moused over
    let currFrameMinClass = Infinity // holding variable for smallest classification number for ship in frame

    for (let i = 0; i < targetShips.length; i++) {
        target = targetShips[i]
        textInfo = target.draw(ctx, aircraft1, aircraft2)
        target.move()

        // RL: set the observation space
        currentObservation["observation"]["threat pos"][i] = [pxToUnaryW(target.x), pxToUnaryH(target.y)]  // target positions
        // min distance to ship
        aircraft1NormalizedDistanceToShip = distance(target.x, aircraft1.x, target.y, aircraft1.y)
        aircraft2NormalizedDistanceToShip = distance(target.x, aircraft2.x, target.y, aircraft2.y)
        if (!target.seenThreatClass && aircraft1NormalizedDistanceToShip < minDistancesToShips[i]) {
            minDistancesToShips[i] = aircraft1NormalizedDistanceToShip
        }
        if (!target.seenThreatClass && aircraft2NormalizedDistanceToShip < minDistancesToShips[i]) {
            minDistancesToShips[i] = aircraft2NormalizedDistanceToShip
        }

        // RL: set whether the target threat class has been seen
        seenTargetClasses[i] = target.seenTargetClass
        seenThreatClasses[i] = target.seenThreatClass

        // target threat class
        if (!target.seenThreatClass) {
            currentObservation["observation"]["threat class"][i] = [1, 0, 0, 0]
        }
        else {
            switch (target.threatClass) {  // threat class is one-hot of the form [unknown, level 1, level 2, level 3]
                case 1: currentObservation["observation"]["threat class"][i] = [0, 1, 0, 0]; break
                case 2: currentObservation["observation"]["threat class"][i] = [0, 0, 1, 0]; break
                case 3: currentObservation["observation"]["threat class"][i] = [0, 0, 0, 1]; break
            }
        }
                

        // populate grid where target moves to
        targetgrid = pxToGrid(target.x, target.y)

        // determine grid block radius
        let dangerRingGridRadius = Math.ceil(target.dangerRadius / gridWidth) + 2
        
        // block out all grid items in that block radius
        if (target.seenThreatClass) {
            for (let i = -1*dangerRingGridRadius; i <= dangerRingGridRadius; i++) {
                for (let j = -1*dangerRingGridRadius; j <= dangerRingGridRadius; j++) {
                    // check if within the ring range
                    if (i * i + j * j < dangerRingGridRadius * dangerRingGridRadius) {
                        // check boundary
                        if (targetgrid.y + j < gridHeight && targetgrid.y + j >= 0 && targetgrid.x + i < gridWidth && targetgrid.y + i >= 0) {
                            grid2d[targetgrid.y + j][targetgrid.x + i] = 0;
                        }
                    }
                }
            }
        }

        mouseOverTarget = target.currentlyMousedOver ? target : mouseOverTarget;  // set the mouse over target        

        // let bearingfromUser = Math.atan2((target.y - aircraft1.y), (target.x - aircraft1.x));
        // if (bearingfromUser < 0) { bearingfromUser += 2*Math.PI; }
        // bearingfromUser += Math.PI / 2; // Canvas rotation starts 0deg as up
        // bearingfromUser %= 2*Math.PI;
        
        // check if target is close to the aircraft
        let distanceToAircraft1 = distance(target.x, aircraft1.x, target.y, aircraft1.y);
        let distanceToAircraft2 = distance(target.x, aircraft2.x, target.y, aircraft2.y);
        let withinDangerRingAircraft1 = (distanceToAircraft1 < target.dangerRadius)
        let withinDangerRingAircraft2 = (distanceToAircraft2 < target.dangerRadius)

        // increment damage score
        if (withinDangerRingAircraft1 == true){ 
            scoreAircraft1 = scoreAircraft1 + 1;
            document.getElementById("score-out").value = scoreAircraft1 + " | " + scoreAircraft2;
        }

        if (withinDangerRingAircraft2 == true){ 
            scoreAircraft2 = scoreAircraft2 + 1;
            document.getElementById("score-out").value = scoreAircraft1 + " | " + scoreAircraft2;
        }

        // update min classification amongst target
        if (target.numClassification < currFrameMinClass){
            currFrameMinClass = target.numClassification
        }

        // Handle alerts for threats
        if (target.targetClass == 1) {
            // Alert to Weapon Employment Zone (WEZ)
            if (!target.isVisible) {
                if (distanceToAircraft1 <= gameWidth*(0.3/2)) {
                    pushAlert("Aircraft 1 Approaching Undetermined WEZ");
                    target.isVisible = true;
                    wez_alerts++;
                }
                if (distanceToAircraft2 <= gameWidth*(0.3/2)) {
                    pushAlert("Aircraft 2 Approaching Undetermined WEZ");
                    target.isVisible = true;
                    wez_alerts++;
                }

                if (target.threatClass == 3 && gameWidth*(0.1/2) < distanceToAircraft1 && distanceToAircraft1 <= gameWidth*(0.3/2)) {
                    pushAlert("Aircraft 1 Approaching Large WEZ");
                    target.isVisible = true;
                    wez_alerts++;
                }
                if (target.threatClass == 3 && gameWidth*(0.1/2) < distanceToAircraft2 && distanceToAircraft2 <= gameWidth*(0.3/2)) {
                    pushAlert("Aircraft 2 Approaching Large WEZ");
                    target.isVisible = true;
                    wez_alerts++;
                }

                if (target.threatClass == 2 && gameWidth*(0.05/2) < distanceToAircraft1 && distanceToAircraft1 <= gameWidth*(0.1/2)) {
                    pushAlert("Aircraft 1 Approaching Medium WEZ");
                    target.isVisible = true;
                    wez_alerts++;
                }
                if (target.threatClass == 2 && gameWidth*(0.05/2) < distanceToAircraft2 && distanceToAircraft2 <= gameWidth*(0.1/2)) {
                    pushAlert("Aircraft 2 Approaching Medium WEZ");
                    target.isVisible = true;
                    wez_alerts++;
                }
                if (target.threatClass == 1 && distanceToAircraft1 <= gameWidth*(0.05/2)) {
                    pushAlert("Aircraft 1 Approaching Small WEZ");
                    target.isVisible = true;
                    wez_alerts++;
                }
                if (target.threatClass == 1 && distanceToAircraft2 <= gameWidth*(0.05/2)) {
                    pushAlert("Aircraft 2 Approaching Small WEZ");
                    target.isVisible = true;
                    wez_alerts++;
                }
            }

            // Check if there is a combatant within 25% gameWidth radius
            if (distanceToAircraft1 < gameWidth*(0.5/2) || distanceToAircraft2 < gameWidth*(0.5/2)) {
                isAllCombatantsInvisible = false;
            }
        }

        // Check if there is a target within 60% gameWidth radius
        if (distanceToAircraft1 < gameWidth*(0.6/2) | distanceToAircraft2 < gameWidth*(0.6/2)) {
            isAllTargetsInvisible = false;
        }
    }

    if (currFrameMinClass > minClassificationCounter){
        minClassificationCounter = currFrameMinClass
    }

    // If AILevel2 and showCollAvoidWaypoints, then run Astar
    result = []
    if (showCollAvoidWaypoints){
        //  run Astar search on occupancy grid, from a/c to next waypoint [AIRCRAFT 1]
        graph = new Graph(grid2d, { diagonal: true });
        start = graph.grid[ac1Grid.y][ac1Grid.x];
        end = graph.grid[ac1NextWPGrid.y][ac1NextWPGrid.x];
        result = astar.search(graph, start, end);
        // if Astar returns results, update collision avoidance waypoint
        if (result.length != 0){
            let pathMidIdx = Math.floor(result.length/2)
            let AstarMidNodePx = gridToPx(result[pathMidIdx].y,result[pathMidIdx].x)
            aircraft1.collAvoidWaypoint.x = Math.floor(AstarMidNodePx.x)
            aircraft1.collAvoidWaypoint.y = Math.floor(AstarMidNodePx.y)
        }

        //  run Astar search on occupancy grid, from a/c to next waypoint [AICRAFT 2]
        graph = new Graph(grid2d, { diagonal: true });
        start = graph.grid[ac2Grid.y][ac2Grid.x];
        end = graph.grid[ac2NextWPGrid.y][ac2NextWPGrid.x];
        result = astar.search(graph, start, end);
        // if Astar returns results, update collision avoidance waypoint
        if (result.length != 0){
            let pathMidIdx = Math.floor(result.length/2)
            let AstarMidNodePx = gridToPx(result[pathMidIdx].y,result[pathMidIdx].x)
            aircraft2.collAvoidWaypoint.x = Math.floor(AstarMidNodePx.x)
            aircraft2.collAvoidWaypoint.y = Math.floor(AstarMidNodePx.y)
        }
    }

    // draw the aircraft, should be after A*
    aircraft1.draw(ctx, result, aircraft1Vector);
    aircraft2.draw(ctx, result, aircraft2Vector);
    aircraft1Vector = aircraft1.move(aircraft1Vector, clickX, clickY, prevAircraft1Vector);
    prevAircraft1Vector = aircraft1Vector
    aircraft2Vector = aircraft2.move(aircraft2Vector, aircraft2VectorX, aircraft2VectorY, prevAircraft2Vector);
    prevAircraft2Vector = aircraft2Vector
    
    // set the observation
    currentObservation["observation"]["agent1 pos"] = [pxToUnaryW(aircraft1.x), pxToUnaryH(aircraft1.y)]
    currentObservation["observation"]["agent2 pos"] = [pxToUnaryW(aircraft2.x), pxToUnaryH(aircraft2.y)]
    
    // trigger the render text
    if (mouseOverTarget) {
        mouseOverTarget.renderText(ctx);
    }

    // All combatants are far
    if (isAllCombatantsInvisible) {
        if (!isAllCombatantsInvisibleSent) {
            //pushAlert("Threats out of range")
            isAllCombatantsInvisibleSent = true;
        }
    } else {
        if (isAllCombatantsInvisibleSent) {
            isAllCombatantsInvisibleSent = false;
        }
    }

    // All targets are far
    if (isAllTargetsInvisible) {
       if (!isAllTargetsInvisibleSent) {
           pushAlert("Update Surface Pic: Targets out of range")
           isAllTargetsInvisibleSent = true;
           usprAlertsCounter++;
       }
    } else {
       if (isAllTargetsInvisibleSent) {
           isAllTargetsInvisibleSent = false;
       }
    }

    // Remove targets that go off the map
    targetShips.forEach((target) => {
        if (target.x < zeroX || target.x > zeroX + gameWidth || target.y < zeroY || target.y > zeroY + gameHeight) {
            targetShips.splice(targetShips.indexOf(target), 1);
        }
    });

    // If vector is requested but not yet accepted, draw in blue
    if (clickX != 945 && clickX != aircraft1.nextWaypoint.x) {
        ctx.beginPath()
        ctx.setLineDash([]);
        ctx.fillStyle = "blue"
        ctx.ellipse(clickX, clickY, toPx(0.01)*userAircraftScale, toPx(0.01)*userAircraftScale, 0, 0, 2*Math.PI)
        ctx.fill()

        ctx.beginPath()
        ctx.strokeStyle = "blue"
        ctx.moveTo(aircraft1.x, aircraft1.y)
        ctx.lineTo(clickX, clickY)
        ctx.stroke()
    }

    // aircraft 1 new waypoint marker
    if (newX != clickX && newX != 945) {
        ctx.beginPath()
        ctx.setLineDash([]);
        ctx.fillStyle = "blue"
        ctx.ellipse(newX, newY, toPx(0.01)*userAircraftScale, toPx(0.01)*userAircraftScale, 0, 0, 2*Math.PI)
        ctx.fill()

        ctx.beginPath()
        ctx.strokeStyle = "blue"
        ctx.moveTo(aircraft1.x, aircraft1.y)
        ctx.lineTo(newX, newY)
        ctx.stroke()
    }

    // Phase 1 : Mouse over all target
    let mousedOverAll = true;
    targetShips.forEach((target) => {
        if (!target.mousedOver) {
            mousedOverAll = false;
        }
    });
    // for RL, terminate when all threats have been observed
    if (usingRL && seenThreatClasses.every(x => x)) {
        currentObservation["terminated"] = true
    }
    // normally, terminate when the user has moused over all
    if (mousedOverAll && !phase1Complete) {
        alert("Phase 1 Complete!")
        pushAlert("Phase 1 Complete!")
        phase1Complete = true;
        currentObservation["terminated"] = true  // RL: end the round
        const endTime = new Date().getTime();
        const time = endTime - countupTimerStart;
        fetch('/saveData', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({"phase": 1, "time": time, "score": scoreAircraft1, "numClicks": numClicks, "unixTime": endTime, "denied": denied, 
                "numOptimized": numOptimized, "wez_alerts": wez_alerts, "osi_alerts": osi_alerts, "tmv_alerts": tmv_alerts, "utc_alerts": utc_alerts})
        })
    }
    // Phase 2: Get target class classification for all ship
    let seenAll = true;
    targetShips.forEach((target) => {
        if (!target.seenTargetClass) {
            seenAll = false;
        }
    });
    if (seenAll && !phase2Complete && phase1Complete) {
        
        pushAlert("Phase 2 Complete!")
        pushAlert("Mission Complete")
        alert("Mission Complete!");
        phase2Complete = true;
        const endTime = new Date().getTime();
        const time = endTime - countupTimerStart;
        fetch('/saveData', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({"phase": 2, "time": time, "score": scoreAircraft1, "numClicks": numClicks, "unixTime": endTime, "denied": denied, 
                "numOptimized": numOptimized, "wez_alerts": wez_alerts, "osi_alerts": osi_alerts, "tmv_alerts": tmv_alerts, "utc_alerts": utc_alerts})
        })
        hasStarted=false;
    }   
    
    // Phase 3: Classifiy each target numClassif times
    if (minClassificationCounter >= numClassificationsToTrack && !phase3Complete && phase1Complete) {
        alert("Mission Complete: Tracked, Classified all targets 2x");
        pushAlert("Phase 3 Complete!")
        pushAlert("Mission Complete")
        phase3Complete = true;
        const endTime = new Date().getTime();
        const time = endTime - startTime;
        fetch('/saveData', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({"phase": 3, "time": time, "score": scoreAircraft1, "numClicks": numClicks})
        })
        hasStarted=false;
    }
}

async function getSimulatorData(){
    // retrieve MSFS a/c location
    if (usingMSFS){
        await fetch("/aircraft")
        .then((resp) => resp.json())
        .then((data) => {
            aircraft1.simx = parseFloat(data["aircraft 1 x"])
            aircraft1.simy = parseFloat(data["aircraft 1 y"])
            aircraft1.simheading = parseFloat(data["aircraft 1 heading"])
            aircraft2.simx = parseFloat(data["aircraft 2 x"])
            aircraft2.simy = parseFloat(data["aircraft 2 y"])
            aircraft2.simheading = parseFloat(data["aircraft 2 heading"])
        })
        .catch((error) => {
            console.log("Failed to pull sim coordinates!!")   
        });
    }
    return false
}

// Optimization functions
function generateOptimization() {
    if(document.getElementById("buttonContainer").style.opacity != "1") return;
    pushAlert('Generated optimization')
    let lastShiftEast = shift_east
    let lastShiftSouth = shift_south
    while (shift_east == lastShiftEast && shift_south == lastShiftSouth) {
        shift_east = Math.round(Math.random()) * 0.4 * gameWidth // (0 or 1) * 1/2 of (0.8*Width)
        shift_south = Math.round(Math.random())* 0.4 * gameHeight // (0 or 1) * 1/2 of (0.8*gameHeight)
    }
    showOptimizedSearchPattern = true;
    numOptimized++;
    aircraft1.optimize(shift_east, shift_south);
    aircraft2.optimize(shift_east, shift_south);
}
function acceptOptimization(){
    if(document.getElementById("buttonContainer").style.opacity != "1") return;
    if (!showOptimizedSearchPattern) {
        pushAlert('Error: Optimized Search not initialized')
        osi_alerts++;
        return;
    }
    pushAlert('Initiated Optimized Search')
    currentlyOptimized = true;
    showOptimizedSearchPattern = false;
    aircraft1.path.waypoints = aircraft1.optimizedPath.waypoints
    aircraft1.path.currentWaypointIndex = 0
    aircraft1.path.currentWaypoint = aircraft1.path.waypoints[0]
    // send initial optimized path aircraft destination
    fetch('/current-destination', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({"aircraft 1 dest x": (aircraft1.path.currentWaypoint.x - zeroX) / gameWidth, "aircraft 1 dest y": (aircraft1.path.currentWaypoint.y - zeroY) /gameHeight})
    })
}
function cancelOptimization(){
    if(document.getElementById("buttonContainer").style.opacity != "1") return;
    if (showOptimizedSearchPattern) {
        showOptimizedSearchPattern = false;
        pushAlert('Rejected Optimized Search')
        return;
    }
    if (!currentlyOptimized) {
        pushAlert('Error: Optimized Search not initialized')
        osi_alerts++;
        return;
    }
    pushAlert('Canceled Optimized Search')
    currentlyOptimized = false;
    aircraft1.path.waypoints = aircraft1.originalPath.waypoints
    aircraft1.path.currentWaypointIndex = 0
    aircraft1.path.currentWaypoint = aircraft1.path.waypoints[0]
    // send initial aircraft path destination
    fetch('/current-destination', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({"aircraft 1 dest x": (aircraft1.path.currentWaypoint.x - zeroX) / gameWidth, "aircraft 1 dest y": (aircraft1.path.currentWaypoint.y - zeroY) /gameHeight})
    })
}
