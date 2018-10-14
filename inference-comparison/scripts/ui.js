const statusElem = document.getElementById("status");

function statusLog(status){
    statusElem.innerText = "Status: " + status; 
}

function triggerStart(){
    let event = new CustomEvent("started", {
        type: "started",
        message: null
    });
    console.log("start");
    document.dispatchEvent(event);
}

function triggerEnd(msg){
    let event = new CustomEvent("finished", {
        type: "finished",
        message: msg
    });
    console.log(msg);
    document.dispatchEvent(event);
}