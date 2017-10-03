function arraysEqual(a, b) {
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; ++i) {
      if (a[i] !== b[i]) return false;
    }
    return true;
}

function updateQueue(identities) {
    if (queue.length > 10) {
       return;
    }
    for (var i = 0; i < queue.length; i++) {
        if (!~identities.indexOf(queue[i])) {
            queue.splice(i, 1);
            i--;
        }
    }
    var minLen = Math.min(identities.length, queue.length)
    if (minLen) {
        if (arraysEqual(identities.slice(-minLen), queue.slice(-minLen))) {
            return;
        }
    }
    queue = queue.concat(identities);
}


