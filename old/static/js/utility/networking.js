async function get(route) {
    const resp = await fetch(route)
    if (!resp.ok) {
        throw new Error('Network response was not ok');
    }
    const json = await resp.json()
    return json
}

async function post(route, data) {
    fetch(route, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
}

// from pixel to [-1, 1]
function pxToUnaryW(val) {
    return (val - zeroX) / canvasLength * 2 - 1
}

// from pixel to [-1, 1]
function pxToUnaryH(val) {
    return (val - zeroY) / canvasLength * 2 - 1
}

// from [-1, 1] to pixel
function unaryToPxW(val) {
    return (val + 1) * canvasLength / 2 + zeroX
}

// from [-1, 1] to pixel
function unaryToPxH(val) {
    return (val + 1) * canvasLength / 2 + zeroY
}