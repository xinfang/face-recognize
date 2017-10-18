function arraysEqual(a, b) {
    if (a === b) return true;
    if (a == null || b == null) return false;
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; ++i) {
      if (a[i] !== b[i]) return false;
    }
    return true;
}

function syncQueue(peoples) {
    if (queue.length > 10) {
       return;
    }
    var result = [];
    for (var i = 0; i < queue.length; i++) {
        var index = 1
        var isInCurrent = peoples.findIndex(function(p){
                 return p.name = queue[i].name
            });
        if (isInCurrent < 0) {
            if (new Date().getTime() - queue[i].time >= 5000) {
                queue.splice(i, 1);
                i--;
            }
       } else {
            queue[i].time = peoples[isInCurrent].time;
       }
    }
    queue.concat(peoples).forEach(item => {
        if (!result.find(function(p){
            return p.name = item.name;
        })) {
            result.push(item);
        }
    });
    queue = result;
}




