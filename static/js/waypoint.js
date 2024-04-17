
class Waypoint {
    constructor(x, y, path) {
        this.x = x;
        this.y = y;
        this.path = path;
        this.timesCompleted = 0;
    }
}

class WaypointPath {
    constructor(points) {
        this.waypoints = [];
        points.forEach(point => {
            this.waypoints.push(new Waypoint(point.x, point.y, this));
        });

        this.currentWaypointIndex = 0;
        this.currentWaypoint = this.waypoints[0];
        this.arrivalRadius = 0.01
        if (usingMSFS){
            this.arrivalRadius = 0.05
        }
    }
    getDirection(x, y, aircraftVector, aircraftVectorX, aircraftVectorY, prevAircraftVector, aircraftName) {
        // User Vector override
        let waypointX = this.currentWaypoint.x;
        let waypointY = this.currentWaypoint.y;

        if (aircraftVector) {
            // if aircraft vector, override AI search pattern
            waypointX = aircraftVectorX;
            waypointY = aircraftVectorY;
            // if new aircraft vector, then move next AI waypoint to next waypoint in search pattern
            if ((prevAircraftVector == false) || ((newX != clickX))){
                // send new destination waypoint to webserver
                if (usingMSFS){
                    fetch('/current-destination', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({"dest x": (waypointX - zeroX) / gameWidth, "dest y": (waypointY - zeroY) / gameWidth})
                    }).catch((error) => {
                        //console.error('Error:', error);
                        console.log("Failed to send sim current destination!!")   
                    });
                }
            }
            this.currentWaypoint = this.waypoints[this.currentWaypointIndex]; 
        }

        // Check if the object has arrived at the current waypoint target
        let distance = Math.sqrt(Math.pow(Math.abs(waypointX - x), 2) + Math.pow(Math.abs(waypointY - y), 2));
        if (distance <= toPx(this.arrivalRadius) && (!usingRL || aircraftName == "aircraft 1")) {  // only run if using aircraft 1, since aircraft 2 is AI controlled
            // Start the next waypoint target, increment if AI waypoint
            if (newX != waypointX) this.currentWaypointIndex++;
                // if at end of waypoint list, reset
                if (this.currentWaypointIndex === this.waypoints.length) {
                    this.timesCompleted++;
                    this.currentWaypointIndex = 0;
                }
                this.currentWaypoint = this.waypoints[this.currentWaypointIndex]; 
            // send new destination waypoint to webserver
            let xTitle = aircraftName + " dest x"
            let yTitle = aircraftName + " dest y"
            if (usingMSFS){
                fetch('/current-destination', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({xTitle: (this.currentWaypoint.x - zeroX) / gameWidth, yTitle: (this.currentWaypoint.y - zeroY) /gameWidth})
                }).catch((error) => {
                //console.error('Error:', error);
                    console.log("Failed to send sim current destination!!")   
                });
            }
            
            // clear the User Vector 
            aircraftVector = false;
        }
        // update previous userAircraftVector
        // Get the angle
        let theta = Math.atan2((waypointY - y), (waypointX - x));
        if (theta < 0) { theta += 2*Math.PI; }
        theta += Math.PI / 2; // Canvas rotation starts 0deg as up
        theta %= 2*Math.PI;
        return [theta, aircraftVector];
    }
}